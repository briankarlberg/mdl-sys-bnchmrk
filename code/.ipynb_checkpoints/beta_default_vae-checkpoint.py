import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Layer, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, losses, optimizers, regularizers
from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import argparse

epochs = 50
batch_size = 50
z_dim = 50
activation = tf.nn.relu
systems_column = "System"
cancer_column = "Cancer_type"
kernel_initializer = tf.keras.initializers.GlorotNormal()
bias_initializer = 'zeros'


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        z_log_var = tf.clip_by_value(z_log_var, -0.1, 0.1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder() -> Model:
    encoder_inputs = Input(shape=(data.shape[1],))
    x = Dense(data.shape[1] / 2, activation=activation)(encoder_inputs)
    x = BatchNormalization()(x)
    # x = Dense(data.shape[1] / 3, activation=activation)(x)
    # x = Dense(data.shape[1] / 4, activation=activation)(x)
    # x = Dense(data.shape[1] / 5, activation=activation)(x)
    x = Dense(z_dim, activation="relu")(x)
    z_mean = Dense(z_dim, name="z_mean")(x)
    z_log_var = Dense(z_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def create_decoder() -> Model:
    decoder_input = Input(shape=(z_dim,))
    x = Dense(z_dim, activation="relu")(decoder_input)
    # x = Dense(data.shape[1] / 5, activation=activation, kernel_regularizer=regularizers.l2(0.001))(x)
    # x = Dense(data.shape[1] / 4, activation=activation, kernel_regularizer=regularizers.l2(0.001))(x)
    # x = Dense(data.shape[1] / 3, activation=activation, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dense(data.shape[1] / 2, activation=activation)(x)
    decoder_outputs = Dense(data.shape[1], activation='sigmoid')(x)
    return Model(decoder_input, decoder_outputs, name="decoder")


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = losses.mean_squared_error(data, reconstruction)
            kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            total_loss = reconstruction_loss + kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": kl_weight
        }


class KLWeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, kl_weight, max_kl_weight, num_epochs):
        super(KLWeightScheduler, self).__init__()
        self.initial_kl_weight = kl_weight
        self.kl_weight = kl_weight
        self.max_kl_weight = max_kl_weight
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the new value for the KL weight
        new_kl_weight = min(self.max_kl_weight, (epoch / self.num_epochs) * self.max_kl_weight)
        kl_weight.assign(new_kl_weight)
        if 4 < epoch < 8:
            kl_weight.assign(0.5)
        if epoch >= 8:
            kl_weight.assign(4.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="The data file. Can be absolute or relative. If used with a relative path, "
                             "make sure its relative to the scripts working directory.")
    parser.add_argument('--output_dir', '-o', type=Path, required=True,
                        help="The output dir where the results should be stored. If it does not exist, it will be created.")
    args = parser.parse_args()

    data_folder = Path(args.data)
    data = pd.read_csv(args.data, sep="\t")

    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    original_output_dir = Path(output_dir, Path(data_folder).stem)
    data_specific_output_dir = original_output_dir
    for i in range(10000):  # Assuming a sensible upper limit to avoid infinite loops
        if not data_specific_output_dir.exists():
            break
        data_specific_output_dir = Path(f"{data_specific_output_dir}_{i + 1}")

    if not data_specific_output_dir.exists():
        data_specific_output_dir.mkdir(parents=True, exist_ok=True)

    systems = data[systems_column].copy()
    cancer = data[cancer_column].copy()
    sample_id = data["improve_sample_id"].copy()

    # Remove the labels and sample_id columns
    data = data.drop(columns=[systems_column, cancer_column, "improve_sample_id"])

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data))

    input_dim = data.shape[1]

    encoder = create_encoder()
    encoder.summary()
    decoder = create_decoder()
    decoder.summary()

    callbacks = []
    early_stop = EarlyStopping(monitor="reconstruction_loss",
                               mode="min", patience=5,
                               restore_best_weights=True)
    callbacks.append(early_stop)

    # add the kl weight scheduler
    kl_weight = K.variable(0.0000)
    callbacks.append(KLWeightScheduler(kl_weight, max_kl_weight=5, num_epochs=epochs))

    vae = VAE(encoder, decoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    vae.compile(optimizer=optimizer)
    vae.fit(scaled_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Correcting batch effects
    _, _, x_test_encoded = vae.encoder.predict(scaled_data)
    x_test_decoded = vae.decoder.predict(x_test_encoded)

    # Save the reconstructed data
    reconstructed_data = pd.DataFrame(x_test_decoded)

    reconstructed_data.insert(0, cancer_column, cancer)
    reconstructed_data.insert(0, systems_column, systems)
    # set sample id as index of the dataframe
    reconstructed_data.index = sample_id
    reconstructed_data.to_csv(Path(data_specific_output_dir, f"{data_folder.stem}_reconstructed_data.tsv"), sep='\t',
                              index=True)

    # Save the latent space
    latent_space = pd.DataFrame(x_test_encoded)
    latent_space.insert(0, cancer_column, cancer)
    latent_space.insert(0, systems_column, systems)
    latent_space.index = sample_id
    latent_space.to_csv(Path(data_specific_output_dir, f"{data_folder.stem}_latent_space.tsv"), sep='\t', index=True)

    latent_space.to_csv(Path(output_dir, f"{data_folder.stem}_latent_space.tsv"), sep='\t', index=True)
