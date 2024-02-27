import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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
        return x

    @tf.custom_gradient
    def grad_reverse(self, x):
        """Applies the gradient reversal during the backward pass."""
        y = tf.identity(x)

        def custom_grad(dy):
            """Multiply the gradient by -alpha."""
            return -self.alpha * dy

        return y, custom_grad

    def compute_output_shape(self, input_shape):
        return input_shape


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


data = pd.read_csv('colon-adeno_transcriptomics_cell-line+CPTAC.tsv', sep='\t')

cancer_labels = data[[cancer_column]]
systems_labels = data[[systems_column]]

data = data.drop(columns=[cancer_column, systems_column, 'improve_sample_id'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

cancer_le = LabelEncoder()
encoded_cancer_labels = cancer_le.fit_transform(cancer_labels)

systems_le = LabelEncoder()
encoded_systems_labels = systems_le.fit_transform(systems_labels)

input_shape = scaled_data.shape[1]

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

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.compile(optimizer='adam',
              loss={'cancer_output': 'binary_crossentropy', 'systems_output': 'binary_crossentropy'},
              metrics={'cancer_output': 'accuracy', 'systems_output': 'accuracy'},
              loss_weights={'cancer_output': 1.0, 'systems_output': -1.0})


history = model.fit(scaled_data, {'cancer_output': encoded_cancer_labels, 'systems_output': encoded_systems_labels},
                    epochs=10)
