import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.losses import MeanSquaredError

systems_column = "System"
cancer_column = "Cancer_type"

latent_space_dim = 50
batch_size = 64
epochs = 100


def contrastive_loss(z1, z2, labels, margin=1.0):
    """
    Compute the contrastive loss given pairs of latent representations and a binary label indicating
    whether the pair is from the same class or not.

    Parameters:
    - z1, z2: The latent representations of pairs of samples (output of the encoder).
    - labels: The binary labels indicating whether the pair is similar (1) or dissimilar (0).
    - margin: The margin enforced between dissimilar pairs.

    Returns:
    - A scalar contrastive loss value.
    """
    # Calculate the euclidean distance between the latent representations
    distances = tf.reduce_sum(tf.square(z1 - z2), axis=1)

    # Cast labels to float32 to match distances data type
    labels = tf.cast(labels, dtype=tf.float32)

    # Calculate the contrastive loss
    similar_loss = labels * distances
    dissimilar_loss = (1 - labels) * tf.maximum(margin - tf.sqrt(distances), 0.0)
    contrastive_loss_value = 0.5 * tf.reduce_mean(similar_loss + dissimilar_loss)

    return contrastive_loss_value


def make_pairs(features, labels):
    pair_features = []
    pair_labels = []

    # Indices of samples for each class
    class_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    # Convert features to a NumPy array if it's a pandas DataFrame
    if isinstance(features, pd.DataFrame):
        features = features.to_numpy()

    # Iterate over class_indices to create pairs
    for class_label, indices in class_indices.items():
        # Positive pairs from the same class
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pair_features.append([features[indices[i]], features[indices[j]]])
                pair_labels.append(1)

        # Negative pairs from different classes
        all_other_indices = np.hstack([idxs for key, idxs in class_indices.items() if key != class_label])
        for i in indices:
            for j in all_other_indices:
                pair_features.append([features[i], features[j]])
                pair_labels.append(0)

    return np.array(pair_features), np.array(pair_labels)


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder model
def create_encoder():
    inputs = Input(shape=(input_dim,))
    x = Dense(input_dim // 2, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)  # Dropout layer for regularization
    z_mean = Dense(latent_space_dim, name="z_mean")(x)
    z_log_var = Dense(latent_space_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    model = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model


# Decoder model
def create_decoder():
    latent_inputs = layers.Input(shape=(latent_space_dim,))
    x = layers.Dense(input_dim // 2, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='relu')(x)
    model = models.Model(latent_inputs, outputs, name='decoder')
    return model


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Initialize the metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.vae_loss_tracker = tf.keras.metrics.Mean(name="vae_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.vae_optimizer = None
        self.disc_optimizer = None

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.total_loss_tracker,
            self.vae_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.contrastive_loss_tracker
        ]

    def compile(self, vae_optimizer, **kwargs):
        super().compile(**kwargs)
        self.vae_optimizer = vae_optimizer

    def train_step(self, data):
        features1 = data[0][:, 0]
        features2 = data[0][:, 1]
        labels = data[1]
        with (tf.GradientTape() as vae_tape, tf.GradientTape() as disc_tape):
            z_mean1, z_log_var1, z1 = self.encoder(features1)
            z_mean2, z_log_var2, z2 = self.encoder(features2)
            reconstruction1 = self.decoder(z1)
            reconstruction2 = self.decoder(z2)

            # Calculate VAE loss components for each feature in the pair
            reconstruction_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(features1,
                                                                                                            reconstruction1) + \
                                  MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(features2,
                                                                                                            reconstruction2)

            kl_loss = -0.5 * (1 + z_log_var1 - tf.square(z_mean1) - tf.exp(z_log_var1)) \
                      - 0.5 * (1 + z_log_var2 - tf.square(z_mean2) - tf.exp(z_log_var2))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Compute contrastive loss using the latent representations and labels
            c_loss = contrastive_loss(z1, z2, labels, 1)
            # Total VAE loss: reconstruction + KL divergence + contrastive loss
            vae_loss = reconstruction_loss + kl_loss

            total_loss = vae_loss + c_loss

        gradients_vae = vae_tape.gradient(total_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.vae_optimizer.apply_gradients(
            zip(gradients_vae, self.encoder.trainable_weights + self.decoder.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.vae_loss_tracker.update_state(vae_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.contrastive_loss_tracker.update_state(c_loss)

        return {
            "total_loss": total_loss,
            "vae_loss": vae_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "contrastive_loss": c_loss
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True,
                        help="The data file. Can be absolute or relative. If used with a relative path, "
                             "make sure its relative to the scripts working directory.")
    parser.add_argument('--output_dir', '-o', type=Path, required=True,
                        help="The output dir where the results should be stored. If it does not exist, it will be created.")
    args = parser.parse_args()

    file_name = Path(args.data).stem
    loaded_data = pd.read_csv(args.data, sep="\t", index_col=0)
    loaded_data.reset_index(drop=True, inplace=True)

    output_dir = Path(args.output_dir, file_name)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Detecting systems...")
    systems = loaded_data[systems_column].unique()

    # check if sys1 is either cell line or HCMI, if not swap
    if systems[0] not in ["cell-line", "hcmi"]:
        systems[0], systems[1] = systems[1], systems[0]

    print(f"System 1: {systems[0]}")
    print(f"System 2: {systems[1]}")

    # select the data
    s1_data = loaded_data[loaded_data[systems_column] == systems[0]]
    s2_data = loaded_data[loaded_data[systems_column] == systems[1]]

    s1_sample_ids = s1_data.index
    s2_sample_ids = s2_data.index

    # detect cancer types
    cancer_types = loaded_data[cancer_column].unique()
    print(f"Detected {len(cancer_types)} cancer types.")

    # select the cancer data for sys1 and sys2
    s1_cancer_data = s1_data[cancer_column]
    s2_cancer_data = s2_data[cancer_column]

    # select the systems data for sys1 and sys2
    s1_system_data = s1_data[systems_column]
    s2_system_data = s2_data[systems_column]

    # drop the cancer and system columns
    s1_data = s1_data.drop(columns=[cancer_column, systems_column])
    s2_data = s2_data.drop(columns=[cancer_column, systems_column])

    # one hot encode cancer and system
    enc = OneHotEncoder(handle_unknown='ignore')

    s1_cancer_enc = pd.get_dummies(s1_cancer_data, prefix='cancer')
    s1_system_enc = pd.get_dummies(s1_system_data, prefix='system')

    s2_cancer_enc = pd.get_dummies(s2_cancer_data, prefix='cancer')
    s2_system_enc = pd.get_dummies(s2_system_data, prefix='system')

    # label encode using sklearn label encoder the cancer and systems columns
    le = LabelEncoder()
    s1_cancer_series_enc = le.fit_transform(s1_cancer_data)
    s1_system_series_enc = le.fit_transform(s1_system_data)

    s2_cancer_series_enc = le.fit_transform(s2_cancer_data)
    s2_system_series_enc = le.fit_transform(s2_system_data)

    # transfer cancer and system to int
    s1_cancer_series_enc = pd.DataFrame(s1_cancer_series_enc.astype(int), index=s1_cancer_data.index)
    s1_system_series_enc = pd.DataFrame(s1_system_series_enc.astype(int), index=s1_system_data.index)

    s2_cancer_series_enc = pd.DataFrame(s2_cancer_series_enc.astype(int), index=s2_cancer_data.index)
    s2_system_series_enc = pd.DataFrame(s2_system_series_enc.astype(int), index=s2_system_data.index)

    s1_data[cancer_column] = s1_cancer_series_enc
    # 1_data[systems_column] = s1_system_series_enc

    s2_data[cancer_column] = s2_cancer_series_enc
    # s2_data[systems_column] = s2_system_series_enc

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    s1_data = pd.DataFrame(scaler.fit_transform(s1_data), columns=s1_data.columns)
    s2_data = pd.DataFrame(scaler.fit_transform(s2_data), columns=s2_data.columns)

    input_dim = s1_data.shape[1]

    encoder = create_encoder()
    decoder = create_decoder()

    paired_features, paired_labels = make_pairs(s1_data, s1_cancer_series_enc)

    # Convert to TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((paired_features, paired_labels))
    # Shuffle and batch the dataset
    train_dataset = train_dataset.shuffle(buffer_size=len(paired_labels)).batch(batch_size)

    # add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience=5)

    vae = VAE(encoder, decoder)
    vae.compile(vae_optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001))

    history = vae.fit(train_dataset, epochs=100, batch_size=batch_size, callbacks=[early_stopping])

    # combine the data
    combined_data = pd.concat([s1_data, s2_data], axis=0)
    combined_cancer_enc = pd.concat([s1_cancer_enc, s2_cancer_enc], axis=0)
    combined_system_enc = pd.concat([s1_system_enc, s2_system_enc], axis=0)
    combined_cancer_series_enc = pd.concat([s1_cancer_series_enc, s2_cancer_series_enc], axis=0)
    combined_system_series_enc = pd.concat([s1_system_series_enc, s2_system_series_enc], axis=0)
    combined_cancers = pd.Series(pd.concat([s1_cancer_data, s2_cancer_data], axis=0))
    combined_systems = pd.Series(pd.concat([s1_system_data, s2_system_data], axis=0))
    combined_sample_ids = pd.concat([pd.Series(s1_sample_ids), pd.Series(s2_sample_ids)], axis=0)

    cancer_data = loaded_data[cancer_column]
    system_data = loaded_data[systems_column]

    # create a latent space embedding
    _, _, latent_space = encoder.predict(combined_data)

    # create a dataframe with the latent space
    z_df = pd.DataFrame(latent_space)
    umap_df = z_df.copy()
    # save latent space
    z_df.insert(0, cancer_column, combined_cancers.values)
    z_df.insert(0, systems_column, combined_systems.values)
    z_df.index = combined_sample_ids.values

    # create latent space file name
    latent_space_file_name = f"{file_name}_{systems[0]}+{systems[1]}.{latent_space_dim}-ltnt-dim_{epochs}-epchs.tsv"

    z_df.to_csv(Path(output_dir, latent_space_file_name), index=True, sep="\t")

    # plot the training history
    plt.plot(history.history['total_loss'])
    plt.plot(history.history['vae_loss'])
    plt.plot(history.history['reconstruction_loss'])
    plt.plot(history.history['kl_loss'])
    plt.plot(history.history['contrastive_loss'])
    plt.title('Contrastive VAE loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # log the y axis
    plt.yscale('log')

    plt.legend(['Total loss', 'VAE loss', 'Reconstruction loss', 'KL loss', 'Contrastive loss'], loc='upper right')
    # save fig
    plt.savefig(Path(output_dir, f"{file_name}_loss_history.png"), dpi=300)
