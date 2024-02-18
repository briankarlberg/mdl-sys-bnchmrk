import keras.optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import metrics, losses, optimizers, regularizers
from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import argparse
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.model_selection import train_test_split

epochs = 1
batch_size = 64
z_dim = 1500
activation = tf.nn.relu
cancer_column = "Cancer_type"
system_column = "System"
max_kl_weight = 1.0
epsilon = 1e-7
kernel_initializer = tf.keras.initializers.GlorotUniform()
bias_initializer = tf.keras.initializers.Zeros()


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder() -> Model:
    encoder_inputs = Input(shape=(data.shape[1],))
    x = Dense(data.shape[1] // 2, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(
        encoder_inputs)
    x = Dense(data.shape[1] // 3, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(data.shape[1] // 4, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(data.shape[1] // 5, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(z_dim, activation="relu")(x)
    z_mean = Dense(z_dim, name="z_mean")(x)
    z_log_var = Dense(z_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def create_decoder() -> Model:
    decoder_input = Input(shape=(z_dim,))
    x = Dense(z_dim, activation="relu")(decoder_input)
    x = Dense(data.shape[1] // 5, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(data.shape[1] // 4, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(data.shape[1] // 3, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    x = Dense(data.shape[1] // 2, activation=activation, kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(x)
    decoder_outputs = Dense(data.shape[1], activation=activation)(x)
    return Model(decoder_input, decoder_outputs, name="decoder")


def build_systems_classifier(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim // 2, activation='relu')(latent_inputs)
    x = Dense(latent_dim // 3, activation='relu')(x)
    x = Dense(latent_dim // 4, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    classifier = Model(latent_inputs, outputs, name="systems_classifier")
    return classifier


def build_cancer_classifier(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim // 2, activation='relu')(latent_inputs)
    x = Dense(latent_dim // 3, activation='relu')(x)
    x = Dense(latent_dim // 4, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    classifier = Model(latent_inputs, outputs, name="cancer_classifier")
    return classifier


def calculate_kl_weight(current_epoch: int):
    return min(max_kl_weight, (current_epoch / epochs) * max_kl_weight)


def print_progress(current_step, total_steps, bar_length=40):
    """
    Prints a progress = indicating the training progress.

    Args:
    - current_step: int, the current step of the loop (starting from 1).
    - total_steps: int, the total number of steps in the loop.
    - bar_length: int, the length of the progress bar in characters.
    """
    progress = current_step / total_steps
    arrow_length = int(round(progress * bar_length))
    bar = '=' * arrow_length + '-' * (bar_length - arrow_length)
    sys.stdout.write(f"\rProgress: [{bar}] {current_step}/{total_steps} ({progress * 100:.2f}%)")
    sys.stdout.flush()


def print_verbose_output():
    print("KL weight")
    tf.print(kl_weight)
    print("Reconstruction loss")
    tf.print(reconstruction_loss)
    print("KL LOSS")
    tf.print(kl_loss)
    print("VAE Loss")
    tf.print(vae_loss)


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, **kwargs):
        # Your forward pass logic here
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=Path, required=True,
                        help="The data file. Can be absolute or relative. If used with a relative path, "
                             "make sure its relative to the scripts working directory.")
    parser.add_argument('--output_dir', '-o', type=Path, required=True,
                        help="The output dir where the results should be stored. If it does not exist, it will be created.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run the script in debug mode")
    args = parser.parse_args()

    data = pd.read_csv(args.data, sep="\t")
    # print([col for col in list(data.columns) if "entr" not in col])
    # input()
    output_dir = args.output_dir
    verbose = args.verbose

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    system_labels = data[system_column].copy()
    # convert labels to binary using a label encoder
    system_le = LabelEncoder()
    encoded_systems_labels = pd.Series(system_le.fit_transform(system_labels))

    cancer_labels = data[cancer_column].copy()
    cancer_le = LabelEncoder()
    encoded_cancer_labels = pd.Series(cancer_le.fit_transform(cancer_labels))

    sample_id = data["improve_sample_id"].copy()

    # Remove the labels and sample_id columns
    data = data.drop(columns=[cancer_column, system_column, "improve_sample_id"])

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data))

    # split data into train / test
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

    pd.DataFrame(train_data, columns=scaled_data.columns).to_csv(Path(output_dir, "train_data.tsv"), index=False)
    pd.DataFrame(test_data, columns=scaled_data.columns).to_csv(Path(output_dir, "test_data.tsv"), index=False)

    # extract the cancer labels based on the index of the train data and the test data
    train_encoded_cancer_labels = encoded_cancer_labels.iloc[train_data.index]
    test_encoded_cancer_labels = encoded_cancer_labels.iloc[test_data.index]
    test_cancer_labels = cancer_labels.iloc[test_data.index]

    train_encoded_system_labels = encoded_systems_labels.iloc[train_data.index]
    test_encoded_system_labels = encoded_systems_labels.iloc[test_data.index]
    test_system_labels = system_labels.iloc[test_data.index]

    input_dim = data.shape[1]

    encoder = create_encoder()
    encoder.summary()
    decoder = create_decoder()
    decoder.summary()

    # add the kl weight scheduler
    kl_weight = K.variable(0.0000)

    # Compile models
    vae = VAE(encoder, decoder)
    # vae.compile(optimizer=optimizers.Adam())

    # Compile models
    systems_classifier = build_systems_classifier(latent_dim=z_dim)
    cancer_classifier = build_cancer_classifier(latent_dim=z_dim)

    # vae.fit(scaled_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_encoded_system_labels, train_encoded_cancer_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.0000000001)
    total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
    cancer_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='cancer_loss')
    system_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='system_Loss')
    kl_loss_metric = tf.keras.metrics.Mean(name='kl_loss')
    vae_loss_metric = tf.keras.metrics.Mean(name='vae_loss')
    reconstruction_loss_metric = tf.keras.metrics.Mean(name='reconstruction_loss')

    training_losses = []
    vae_weights_history = []
    systems_weight_history = []
    cancer_weight_history = []
    best_accuracy = {"Cancer Accuracy": 0, "Systems Accuracy": 100}

    for epoch in range(epochs):
        print(f"\nStart of epoch: {epoch}")
        total_loss_metric.reset_states()
        cancer_accuracy_metric.reset_states()
        system_accuracy_metric.reset_states()
        kl_loss_metric.reset_states()
        vae_loss_metric.reset_states()
        reconstruction_loss_metric.reset_states()

        for step, (x_batch_train, systems_batch_train, cancers_batch_train) in enumerate(train_dataset):
            if verbose:
                print(f"Step {step}")
            with tf.GradientTape() as tape:
                predictions = vae(x_batch_train, training=True)
                z_mean, z_log_var, z = vae.encoder(x_batch_train)
                reconstruction = vae.decoder(z)
                reconstruction_loss = losses.mean_squared_error(x_batch_train, reconstruction)
                kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
                vae_loss = reconstruction_loss + kl_loss

                if verbose:
                    print_verbose_output()

                # Auxiliary classifier loss
                systems_predictions = systems_classifier(z)
                systems_predictions = tf.squeeze(systems_predictions, axis=-1)

                cancer_predictions = cancer_classifier(z)
                cancer_predictions = tf.squeeze(cancer_predictions, axis=-1)

                systems_classifier_loss = losses.binary_crossentropy(systems_batch_train, systems_predictions)
                cancer_classifier_loss = losses.binary_crossentropy(cancers_batch_train, cancer_predictions)

                # Total loss. Might have to adjust the loss weights
                total_loss = vae_loss + (2 * cancer_classifier_loss) + 1 / (systems_classifier_loss + epsilon)

                # Update the metrics
                total_loss_metric.update_state(total_loss)
                cancer_accuracy_metric.update_state(cancers_batch_train, cancer_predictions)
                system_accuracy_metric.update_state(systems_batch_train, systems_predictions)
                kl_loss_metric.update_state(kl_loss)
                vae_loss_metric.update_state(vae_loss)
                reconstruction_loss_metric.update_state(reconstruction_loss)

            grads = tape.gradient(total_loss,
                                  vae.trainable_weights + cancer_classifier.trainable_weights + systems_classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads,
                                          vae.trainable_weights + cancer_classifier.trainable_weights + systems_classifier.trainable_weights))

            print_progress(step, len(scaled_data) // batch_size)

        if epoch != 0:
            kl_weight = calculate_kl_weight(epoch)

        total_loss_value = total_loss_metric.result()
        cancer_accuracy_value = cancer_accuracy_metric.result()
        systems_accuracy_value = system_accuracy_metric.result()
        kl_loss_value = kl_loss_metric.result()
        vae_loss_value = vae_loss_metric.result()
        reconstruction_loss_value = reconstruction_loss_metric.result()

        print(
            f"\nTotal Loss: {total_loss_value.numpy()} - Cancer Accuracy: {cancer_accuracy_value.numpy()} - Systems Accuracy:"
            f" {systems_accuracy_value.numpy()} - KL Loss: {kl_loss_value.numpy()} - VAE Loss: {vae_loss_value.numpy()} -"
            f" Reconstruction Loss: {reconstruction_loss_value.numpy()}")

        # compare the accuracy of the cancer and systems classifier and save the weights only if the cancer accuracy value is up and the system accuracy value change only 10% compared to the saved value
        if epoch != 0:
            # Calculate the 10% difference thresholds for systems accuracy
            lower_threshold = best_accuracy["Systems Accuracy"] * 0.9
            upper_threshold = best_accuracy["Systems Accuracy"] * 1.1

            if (cancer_accuracy_value.numpy() > best_accuracy["Cancer Accuracy"]
                    and systems_accuracy_value.numpy() <= upper_threshold):
                print(f"Previous Cancer Accuracy: {best_accuracy['Cancer Accuracy']}")
                print(f"Previous Systems Accuracy: {best_accuracy['Systems Accuracy']}")
                print("Updating weights...")

                vae_weights_history = vae.get_weights()
                systems_weight_history = systems_classifier.get_weights()
                cancer_weight_history = cancer_classifier.get_weights()
                best_accuracy["Cancer Accuracy"] = cancer_accuracy_value.numpy()
                best_accuracy["Systems Accuracy"] = systems_accuracy_value.numpy()
        else:
            print("Saved initial weights...")
            vae_weights_history = vae.get_weights()
            systems_weight_history = systems_classifier.get_weights()
            cancer_weight_history = cancer_classifier.get_weights()
            best_accuracy["Cancer Accuracy"] = cancer_accuracy_value.numpy()
            best_accuracy["Systems Accuracy"] = systems_accuracy_value.numpy()

        # add losses to the training losses dict
        training_losses.append({
            "Epoch": epoch,
            "Total Loss": total_loss_value.numpy(),
            "Cancer Accuracy": cancer_accuracy_value.numpy(),
            "Systems Accuracy": systems_accuracy_value.numpy(),
            "KL Loss": kl_loss_value.numpy(),
            "VAE Loss": vae_loss_value.numpy(),
            "Reconstruction Loss": reconstruction_loss_value.numpy()
        })

    print("Training done.")
    # save training loss history
    training_losses = pd.DataFrame(training_losses)
    training_losses.to_csv(Path(output_dir, "training_losses.tsv"), index=False)

    # restore models to the best weights
    vae.set_weights(vae_weights_history)

    systems_classifier.set_weights(systems_weight_history)
    cancer_classifier.set_weights(cancer_weight_history)

    # Save the models
    vae.save(Path(output_dir, "vae"))
    systems_classifier.save(Path(output_dir, "systems_classifier"))
    cancer_classifier.save(Path(output_dir, "cancer_classifier"))

    # Correcting batch effects
    _, _, x_test_encoded = vae.encoder.predict(test_data)
    x_test_decoded = vae.decoder.predict(x_test_encoded)

    # Save the reconstructed data
    reconstructed_data = pd.DataFrame(x_test_decoded)
    reconstructed_data["System"] = system_labels
    reconstructed_data["Cancer"] = cancer_labels
    reconstructed_data["improve_sample_id"] = sample_id
    reconstructed_data.to_csv(Path(output_dir, "reconstructed_data.tsv"), index=False)

    # Save the latent space
    latent_space = pd.DataFrame(x_test_encoded)
    latent_space["Cancer"] = cancer_labels
    latent_space["System"] = system_labels
    latent_space["improve_sample_id"] = sample_id
    latent_space.to_csv(Path(output_dir, "latent_space.tsv"), index=False)

    # predict cancer and system labels
    cancer_predictions = cancer_classifier.predict(x_test_encoded)
    system_predictions = systems_classifier.predict(x_test_encoded)

    # save the predictions
    cancer_predictions = pd.DataFrame(cancer_predictions)
    # use the test set cancer labels
    cancer_predictions["Cancer"] = test_cancer_labels.reset_index(drop=True)
    cancer_predictions.to_csv(Path(output_dir, "cancer_predictions.tsv"), index=False)

    system_predictions = pd.DataFrame(system_predictions)
    system_predictions["System"] = test_system_labels.reset_index(drop=True)
    system_predictions.to_csv(Path(output_dir, "system_predictions.tsv"), index=False)
