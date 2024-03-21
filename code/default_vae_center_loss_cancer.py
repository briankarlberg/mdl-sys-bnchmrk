import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, losses
from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import argparse

epochs = 50
batch_size = 64
z_dim = 50
activation = tf.nn.relu
systems_column = "System"
cancer_column = "Cancer_type"

class CenterLossLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = tf.Variable(initial_value=tf.zeros((num_classes, feature_dim)),
                                   trainable=False,
                                   dtype=tf.float32)

    def call(self, inputs, labels):
        # Get the centers of the corresponding labels
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        centers_batch = tf.gather(self.centers, labels)
        # Update centers
        diff = centers_batch - inputs
        unique_labels, unique_idx, counts = tf.unique_with_counts(labels)
        counts = tf.cast(counts, tf.float32)
        diff_sum = tf.math.unsorted_segment_sum(diff, unique_idx, tf.size(unique_labels))
        centers_update = tf.scatter_nd(unique_idx[:, None], diff_sum, tf.shape(self.centers))
        self.centers.assign(self.centers - centers_update / (counts[:, None] + 1))
        # Compute the center loss
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - centers_batch), axis=1))
        self.add_loss(loss)
        return inputs  # Pass-through layer

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "feature_dim": self.feature_dim})
        return config

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(input_dim: int, num_classes: int) -> Model:
    encoder_inputs = Input(shape=(input_dim,))
    labels = Input(shape=(num_classes,))
    # add normalization layer
    # x = Normalization(axis=-1)(encoder_inputs)
    x = Concatenate()([encoder_inputs, labels])
    x = Dense(input_dim // 2, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dense(z_dim, activation=activation)(x)
    z_mean = Dense(z_dim, name="z_mean")(x)
    z_log_var = Dense(z_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model([encoder_inputs, labels], [z_mean, z_log_var, z], name="encoder")


def create_decoder(input_dim, num_classes) -> Model:
    decoder_input = Input(shape=(z_dim,))
    labels = Input(shape=(num_classes,))
    x = Concatenate()([decoder_input, labels])
    x = Dense(z_dim, activation=activation, name='encoder_dense_1')(x)
    x = Dense(input_dim // 2, activation=activation, name='encoder_dense_2')(x)
    decoder_outputs = Dense(input_dim, activation='sigmoid')(x)
    return Model([decoder_input, labels], decoder_outputs, name="decoder")


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
        (features, labels), = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([features, labels])
            reconstruction = self.decoder([z, labels])
            reconstruction_loss = losses.mean_squared_error(features, reconstruction)
            kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)

            center_loss_layer = CenterLossLayer(num_classes=len(unique_cancers), feature_dim=z_dim)
            z_centered = center_loss_layer(z, labels)
            center_loss = sum(center_loss_layer.losses)
            total_loss = reconstruction_loss + kl_weight * kl_loss + center_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": kl_weight,
            "center_loss": center_loss
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
        new_kl_weight = min(self.max_kl_weight, (epoch / self.num_epochs) * self.max_kl_weight) * 100
        if new_kl_weight > 1:
            new_kl_weight = 1
        kl_weight.assign(new_kl_weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True,
                        help="The data file. Can be absolute or relative. If used with a relative path, "
                             "make sure its relative to the scripts working directory.", nargs='+')
    parser.add_argument('--output_dir', '-o', type=Path, required=True,
                        help="The output dir where the results should be stored. If it does not exist, it will be created.")
    args = parser.parse_args()

    dfs = []
    for file in args.data:
        dfs.append(pd.read_csv(file, sep="\t"))

    # systems identifier is the last part of the file name
    systems_identifier = Path(args.data[0]).stem.split("_")[-1]
    file_names = [Path(file).stem for file in args.data]
    # split by _ and extract the last element
    file_names = ['_'.join(file.split("_")[:-1]) for file in file_names]

    data_folder = Path('_'.join(file_names) + f'_{systems_identifier}')

    loaded_data = pd.concat(dfs)
    loaded_data.reset_index(drop=True, inplace=True)

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

    unique_systems = loaded_data[systems_column].unique()
    unique_cancers = loaded_data[cancer_column].unique()

    print("Detecting systems...")
    system_1 = loaded_data[systems_column].unique()[0]
    system_2 = loaded_data[systems_column].unique()[1]

    print(f"System 1: {system_1}")
    print(f"System 2: {system_2}")

    # Filter the data for the two systems
    data_system_1 = loaded_data[loaded_data[systems_column] == system_1]
    data_system_2 = loaded_data[loaded_data[systems_column] == system_2]

    data_system_1_cancer = data_system_1[cancer_column]
    data_system_2_cancer = data_system_2[cancer_column]

    # label encode the cancer types use sklearn label encoder
    le = LabelEncoder()
    data_system_1_cancer_enc = pd.Series(le.fit_transform(data_system_1_cancer))
    data_system_2_cancer_enc = pd.Series(le.fit_transform(data_system_2_cancer))

    data_system_1_sample_ids = data_system_1["improve_sample_id"]
    data_system_2_sample_ids = data_system_2["improve_sample_id"]

    data_systems_1_system = data_system_1[systems_column]
    data_systems_2_system = data_system_2[systems_column]

    # label encode the systems
    le = LabelEncoder()
    data_systems_1_system_enc = pd.Series(le.fit_transform(data_systems_1_system))
    data_systems_2_system_enc = pd.Series(le.fit_transform(data_systems_2_system))

    # Remove the labels and sample_id columns
    data_system_1 = data_system_1.drop(columns=[systems_column, cancer_column, "improve_sample_id"])
    data_system_2 = data_system_2.drop(columns=[systems_column, cancer_column, "improve_sample_id"])

    # scale the data using min max scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_system_1 = pd.DataFrame(scaler.fit_transform(data_system_1))
    data_system_2 = pd.DataFrame(scaler.fit_transform(data_system_2))

    input_dim = data_system_1.shape[1]

    encoder = create_encoder(input_dim=input_dim, num_classes=1)
    encoder.summary()
    decoder = create_decoder(input_dim=input_dim, num_classes=1)
    decoder.summary()

    # add the kl weight scheduler
    kl_weight = K.variable(0.0000)

    callbacks = []
    early_stop = EarlyStopping(monitor="reconstruction_loss",
                               mode="min", patience=5,
                               restore_best_weights=True)
    callbacks.append(early_stop)
    callbacks.append(
        KLWeightScheduler(kl_weight, max_kl_weight=1, num_epochs=epochs))

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001))
    history = vae.fit([data_system_1, data_system_1_cancer_enc], epochs=epochs, batch_size=batch_size,
                      callbacks=callbacks)

    data_combined = pd.concat([data_system_1, data_system_2])
    data_combined_systems = pd.Series(pd.concat([data_systems_1_system, data_systems_2_system]))
    data_combined_systems_enc = pd.Series(pd.concat([data_systems_1_system_enc, data_systems_2_system_enc]))
    data_combined_cancer_enc = pd.Series(pd.concat([data_system_1_cancer_enc, data_system_2_cancer_enc]))
    data_combined_cancer = pd.Series(pd.concat([data_system_1_cancer, data_system_2_cancer]))
    data_combined_sample_ids = pd.concat([data_system_1_sample_ids, data_system_2_sample_ids])

    # Correcting batch effects
    _, _, x_test_encoded = vae.encoder.predict([data_combined, data_combined_cancer_enc])
    x_test_decoded = vae.decoder.predict([x_test_encoded, data_combined_cancer_enc])

    # Save the reconstructed data
    reconstructed_data = pd.DataFrame(x_test_decoded)

    reconstructed_data.insert(0, cancer_column, data_combined_cancer)
    reconstructed_data.insert(0, systems_column, data_combined_systems)
    # set sample id as index of the dataframe
    reconstructed_data.index = data_combined_sample_ids
    reconstructed_data.to_csv(Path(data_specific_output_dir, f"{data_folder.stem}.reconstructed_data.tsv"), sep='\t',
                              index=True)

    # Save the latent space
    latent_space = pd.DataFrame(x_test_encoded)
    latent_space.insert(0, cancer_column, data_combined_cancer)
    latent_space.insert(0, systems_column, data_combined_systems)
    latent_space.index = data_combined_sample_ids
    latent_space.to_csv(Path(data_specific_output_dir, f"{data_folder.stem}.latent_space.tsv"), sep='\t', index=True)

    latent_space.to_csv(Path(output_dir, f"{data_folder.stem}.latent_space.tsv"), sep='\t', index=True)

    # calculate the reconstruction loss
    reconstruction_loss = losses.mean_squared_error(data_combined, x_test_decoded)
    reconstruction_loss = K.mean(reconstruction_loss)
    print(f"Reconstruction loss: {reconstruction_loss}")

    # Save the loss history
    loss_history = pd.DataFrame(history.history)
    loss_history.to_csv(Path(data_specific_output_dir, f"{data_folder.stem}.loss_history.tsv"), sep='\t', index=True)
