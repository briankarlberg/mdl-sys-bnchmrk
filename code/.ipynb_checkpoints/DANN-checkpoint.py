import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split

systems_column = "System"
cancer_column = "Cancer_type"


class GradientReversalLayer(Layer):
    """
    Gradient Reversal Layer.
    During the forward pass, this layer acts as an identity transform.
    During the backward pass, it multiplies the gradient by a negative factor.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, x, mask=None):
        """Performs the identity operation during the forward pass."""
        return self.grad_reverse(x)  # This line is changed to call grad_reverse

    @tf.custom_gradient
    def grad_reverse(self, x):
        """Applies the gradient reversal during the backward pass."""
        y = tf.identity(x)

        def custom_grad(dy):
            """Multiply the gradient by -alpha."""
            return -self.alpha * dy

        return y, custom_grad


def create_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    return Model(inputs, x, name="feature_extractor")


def create_cancer_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    cancer_output = Dense(1, activation='sigmoid', name="cancer_output")(x)  # Adjust based on your classification task
    return Model(inputs, cancer_output, name="cancer_classifier")


def create_systems_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    system_output = Dense(1, activation='sigmoid', name="systems_output")(x)  # Assuming binary domain classification
    return Model(inputs, system_output, name="systems_classifier")


file = Path('../output/colon-adeno_transcriptomics_cell-line+CPTAC.tsv')
data = pd.read_csv(str(file), sep='\t')

# split the data  into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True,
                                         stratify=data[[cancer_column, systems_column]])

train_cancer_labels = train_data[[cancer_column]]
train_systems_labels = train_data[[systems_column]]
train_sample_ids = train_data['improve_sample_id']

test_cancer_labels = test_data[[cancer_column]]
test_systems_labels = test_data[[systems_column]]
test_sample_ids = test_data['improve_sample_id']

train_data = train_data.drop(columns=[cancer_column, systems_column, 'improve_sample_id'])
test_data = test_data.drop(columns=[cancer_column, systems_column, 'improve_sample_id'])
scaled_data = data.drop(columns=[cancer_column, systems_column, 'improve_sample_id'])

scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)
scaled_data = scaler.fit_transform(scaled_data)

cancer_le = LabelEncoder()
encoded_train_cancer_labels = cancer_le.fit_transform(train_cancer_labels)
encoded_test_cancer_labels = cancer_le.transform(test_cancer_labels)
encoded_cancer_labels = cancer_le.transform(data[cancer_column])

systems_le = LabelEncoder()
encoded_train_systems_labels = systems_le.fit_transform(train_systems_labels)
encoded_test_systems_labels = systems_le.transform(test_systems_labels)
encoded_systems_labels = systems_le.transform(data[systems_column])

input_shape = train_data.shape[1]

# Assuming input data shape is (100,)
feature_extractor = create_feature_extractor(input_shape=(input_shape,))
feature_output = feature_extractor.output

# Apply the Gradient Reversal Layer
grl_output = GradientReversalLayer(alpha=1.0)(feature_output)

# Create the classifier
cancer_classifier = create_cancer_classifier(input_shape=(feature_output.shape[1],))
systems_classifier = create_systems_classifier(input_shape=(feature_output.shape[1],))

# Get the models output
cancer_output = cancer_classifier(feature_output)
systems_output = systems_classifier(grl_output)

cancer_output._name = 'cancer_output'
systems_output._name = 'systems_output'

# Complete model with two outputs
model_input = feature_extractor.input
model = Model(inputs=model_input, outputs=[cancer_output, systems_output])

model.summary()
systems_classifier.summary()
cancer_classifier.summary()

# add early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='systems_classifier_accuracy', patience=5)

model.compile(optimizer='adam',
              loss={'cancer_classifier': 'binary_crossentropy', 'systems_classifier': 'binary_crossentropy'},
              metrics={'cancer_classifier': 'accuracy', 'systems_classifier': 'accuracy'},
              loss_weights={'cancer_classifier': 1.0, 'systems_classifier': 10.0})

history = model.fit(scaled_train_data,
                    {'cancer_classifier': encoded_train_cancer_labels,
                     'systems_classifier': encoded_train_systems_labels},
                    epochs=100, callbacks=[early_stop], validation_split=0.2)

domain_invariant_features = feature_extractor.predict(scaled_data)
# create a feature invariant representation by using the feature extractor
domain_invariant_features_df = pd.DataFrame(domain_invariant_features)
domain_invariant_features_df.insert(0, cancer_column, encoded_cancer_labels)
domain_invariant_features_df.insert(0, systems_column, encoded_systems_labels)
domain_invariant_features_df.index = data['improve_sample_id']

# insert the cancer and system labels


domain_invariant_features_df.to_csv(Path(f"{file.stem}_latent_space.tsv"), sep='\t', index=True)

# save history
history = pd.DataFrame(history.history)
history.to_csv(Path(f"{file.stem}_history.tsv"), sep='\t', index=True)
