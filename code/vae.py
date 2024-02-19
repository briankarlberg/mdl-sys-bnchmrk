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
from sklearn.metrics import f1_score, precision_score, recall_score

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
target_cancer_accuracy = 0.9
target_systems_accuracy = 0.2


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


def calculate_systems_entropy_penalty(systems_classifier_predictions, entropy_epsilon=1e-7):
    systems_classifier_predictions = tf.clip_by_value(systems_classifier_predictions, entropy_epsilon,
                                                      1 - entropy_epsilon)
    entropy = -(systems_classifier_predictions * tf.math.log(systems_classifier_predictions) +
                (1 - systems_classifier_predictions) * tf.math.log(1 - systems_classifier_predictions))
    penalty = -tf.reduce_mean(entropy)  # We want to minimize this loss, hence negative entropy
    return penalty


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


def adjust_loss_weights(cancer_accuracy, systems_accuracy):
    global cancer_weight_multiplier, systems_weight_multiplier

    # if cancer_accuracy >= target_cancer_accuracy:
    #    # Prioritize systems loss to decrease systems accuracy
    #    systems_weight_multiplier = 6.0  # increase weight
    #    cancer_weight_multiplier = 2.0  # Decrease weight to maintain cancer accuracy
    # elif systems_accuracy < target_systems_accuracy and cancer_accuracy < target_cancer_accuracy:
    #    # Prioritize cancer loss to improve cancer accuracy
    #    cancer_weight_multiplier = 4.0  # Example: increase weight
    #    systems_weight_multiplier = 2.0  # Decrease weight to deprioritize systems accuracy
    # else:
    # Reset to default weights if none of the conditions are met
    #    cancer_weight_multiplier = 2.0
    #    systems_weight_multiplier = 2.0


def calculate_total_loss():
    # systems_loss_penalty = -tf.math.log(systems_classifier_loss + epsilon)
    # scaled_systems_loss_penalty = systems_weight_multiplier * systems_loss_penalty
    scaled_cancer_loss = cancer_weight_multiplier * cancer_classifier_loss
    systems_entropy = calculate_systems_entropy_penalty(systems_predictions)

    # Apply softplus as systems loss penalty transformation
    # Note: Adjusting the expression inside softplus based on your specific needs
    scaled_systems_loss_penalty = tf.nn.softplus(systems_weight_multiplier * (1 / (systems_classifier_loss + epsilon)))

    return vae_loss + scaled_cancer_loss + scaled_systems_loss_penalty + systems_entropy


def save_model_weights(vae, systems_classifier, cancer_classifier, best_accuracy, cancer_accuracy_value,
                       systems_accuracy_value):
    # Define thresholds for acceptable changes
    cancer_accuracy_improvement_threshold = 0.03
    systems_accuracy_improvement_threshold = 0.01
    cancer_accuracy_allowable_decrease = 0.01
    systems_accuracy_significant_decrease = 0.03

    # Calculate the changes in accuracy
    change_in_cancer_accuracy = cancer_accuracy_value.numpy() - best_accuracy["Cancer Accuracy"]
    change_in_systems_accuracy = systems_accuracy_value.numpy() - best_accuracy["Systems Accuracy"]

    # Check conditions for updating weights
    if ((change_in_cancer_accuracy >= cancer_accuracy_improvement_threshold and
         change_in_systems_accuracy <= systems_accuracy_improvement_threshold) or
            (change_in_cancer_accuracy > -cancer_accuracy_allowable_decrease and
             change_in_systems_accuracy <= -systems_accuracy_significant_decrease)):
        print(f"Previous Cancer Accuracy: {best_accuracy['Cancer Accuracy']}")
        print(f"Previous Systems Accuracy: {best_accuracy['Systems Accuracy']}")
        print("Updating weights...")

        best_accuracy["Cancer Accuracy"] = cancer_accuracy_value.numpy()
        best_accuracy["Systems Accuracy"] = systems_accuracy_value.numpy()
        best_accuracy["Epoch"] = epoch

        return vae.get_weights(), systems_classifier.get_weights(), cancer_classifier.get_weights()
    else:
        return vae_weights, systems_classifier_weights, cancer_classifier_weights


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
    parser.add_argument("--epochs", "-e", action="store", type=int, default=50,
                        help="The amount of epochs to train the model")
    parser.add_argument("--batch_size", "-b", action="store", type=int, default=64,
                        help="The batch size to use for training")
    parser.add_argument("--latent_space", "-lts", action="store", type=int, default=1500,
                        help="The latent space dimension to use for the VAE")
    parser.add_argument("--cancer_multiplier", "-cm", action="store", type=float, default=2.0,
                        help="The multiplier to use for the cancer classifier loss")
    parser.add_argument("--systems_multiplier", "-sm", action="store", type=float, default=2.5,
                        help="The multiplier to use for the systems classifier loss")
    args = parser.parse_args()

    data_folder: Path = Path(args.data)
    data = pd.read_csv(args.data, sep="\t")
    # print([col for col in list(data.columns) if "entr" not in col])
    # input()
    output_dir: Path = args.output_dir
    verbose: bool = args.verbose
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    z_dim: int = args.latent_space
    cancer_weight_multiplier = tf.constant(args.cancer_multiplier)
    systems_weight_multiplier = tf.constant(args.systems_multiplier)

    original_dir = Path(output_dir, Path(data_folder).stem)
    output_dir = original_dir
    for i in range(10000):  # Assuming a sensible upper limit to avoid infinite loops
        if not output_dir.exists():
            break
        output_dir = Path(f"{original_dir}_{i + 1}")

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

    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.00000001)
    total_loss_metric = tf.keras.metrics.Mean(name="total_loss")
    cancer_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='cancer_loss')
    system_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='system_Loss')
    kl_loss_metric = tf.keras.metrics.Mean(name='kl_loss')
    vae_loss_metric = tf.keras.metrics.Mean(name='vae_loss')
    reconstruction_loss_metric = tf.keras.metrics.Mean(name='reconstruction_loss')

    training_losses = []
    vae_weights = []
    systems_weight = []
    cancer_weight = []
    best_accuracy = {"Cancer Accuracy": 0, "Systems Accuracy": 100, "Epoch": 0}

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
                _ = vae(x_batch_train, training=True)
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

                cancer_accuracy_metric.update_state(cancers_batch_train, cancer_predictions)
                system_accuracy_metric.update_state(systems_batch_train, systems_predictions)

                # Total loss. Might have to adjust the loss weights
                # total_loss = vae_loss + (2 * cancer_classifier_loss) + 1 / (systems_classifier_loss + epsilon)

                adjust_loss_weights(cancer_accuracy_metric.result(), system_accuracy_metric.result())
                total_loss = calculate_total_loss()

                # Update the metrics
                total_loss_metric.update_state(total_loss)

                kl_loss_metric.update_state(kl_loss)
                vae_loss_metric.update_state(vae_loss)
                reconstruction_loss_metric.update_state(reconstruction_loss)

            grads = tape.gradient(total_loss,
                                  vae.trainable_weights + cancer_classifier.trainable_weights + systems_classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads,
                                          vae.trainable_weights + cancer_classifier.trainable_weights + systems_classifier.trainable_weights))

            print_progress(step, len(train_data) // batch_size)

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
            vae_weights, systems_classifier_weights, cancer_classifier_weights = save_model_weights(vae,
                                                                                                    systems_classifier,
                                                                                                    cancer_classifier,
                                                                                                    best_accuracy,
                                                                                                    cancer_accuracy_value,
                                                                                                    systems_accuracy_value)


        else:
            print("Saved initial weights...")
            vae_weights = vae.get_weights()
            systems_classifier_weights = systems_classifier.get_weights()
            cancer_classifier_weights = cancer_classifier.get_weights()
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
    vae.set_weights(vae_weights)

    systems_classifier.set_weights(systems_classifier_weights)
    cancer_classifier.set_weights(cancer_classifier_weights)

    # Save the models
    vae.save(Path(output_dir, "vae"))
    systems_classifier.save(Path(output_dir, "systems_classifier"))
    cancer_classifier.save(Path(output_dir, "cancer_classifier"))

    # Correcting batch effects
    _, _, x_test_encoded = vae.encoder.predict(test_data)
    x_test_decoded = vae.decoder.predict(x_test_encoded)

    # Save the reconstructed data
    reconstructed_data = pd.DataFrame(x_test_decoded)
    reconstructed_data["System"] = system_labels.reset_index(drop=True)
    reconstructed_data["Cancer"] = cancer_labels.reset_index(drop=True)
    reconstructed_data.to_csv(Path(output_dir, "reconstructed_data.tsv"), index=False)

    # Save the latent space
    latent_space = pd.DataFrame(x_test_encoded)
    latent_space["Cancer"] = cancer_labels.reset_index(drop=True)
    latent_space["System"] = system_labels.reset_index(drop=True)
    latent_space.to_csv(Path(output_dir, "latent_space.tsv"), index=False)

    # predict cancer and system labels
    cancer_predictions = cancer_classifier.predict(x_test_encoded)
    system_predictions = systems_classifier.predict(x_test_encoded)



    run_information = [{
        "File Name": Path(args.data).stem,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Latent Space": z_dim,
        "Epsilon": epsilon,
        "Cancer Weight Multiplier": cancer_weight_multiplier,
        "Systems Weight Multiplier": systems_weight_multiplier,
        "Target Cancer Accuracy": target_cancer_accuracy,
        "Target Systems Accuracy": target_systems_accuracy,
        "Best Cancer Accuracy": best_accuracy["Cancer Accuracy"],
        "Best Systems Accuracy": best_accuracy["Systems Accuracy"],
        "Best Epoch": best_accuracy["Epoch"]
    }]
    run_information = pd.DataFrame(run_information)
    run_information.to_csv(Path(output_dir, "run_information.tsv"), index=False)

    # calculate f1 for cancer predictions using sklearn
    cancer_predictions = np.where(cancer_predictions > 0.5, 1, 0)
    cancer_f1 = f1_score(test_encoded_cancer_labels, cancer_predictions)

    # calculate f1 for system predictions using sklearn
    system_predictions = np.where(system_predictions > 0.5, 1, 0)
    system_f1 = f1_score(test_encoded_system_labels, system_predictions)

    # calculate precision and recall for cancer and system predictions using sklearn
    cancer_precision = precision_score(test_encoded_cancer_labels, cancer_predictions)
    cancer_recall = recall_score(test_encoded_cancer_labels, cancer_predictions)

    system_precision = precision_score(test_encoded_system_labels, system_predictions)
    system_recall = recall_score(test_encoded_system_labels, system_predictions)

    # save the predictions
    cancer_predictions_df = pd.DataFrame(cancer_predictions)
    cancer_predictions_df["Encoded Labels"] = test_encoded_cancer_labels.reset_index(drop=True)
    cancer_predictions_df["Correct Labels"] = test_cancer_labels.reset_index(drop=True)
    cancer_predictions_df["Decoded Predictions"] = cancer_le.inverse_transform(cancer_predictions)
    cancer_predictions_df.to_csv(Path(output_dir, "cancer_predictions.tsv"), index=False)

    system_predictions_df = pd.DataFrame(system_predictions)
    system_predictions_df["Encoded Labels"] = test_encoded_system_labels.reset_index(drop=True)
    system_predictions_df["System"] = test_system_labels.reset_index(drop=True)
    system_predictions_df["Decoded Predictions"] = system_le.inverse_transform(system_predictions)
    system_predictions_df.to_csv(Path(output_dir, "system_predictions.tsv"), index=False)

    # create a df with f1, precision and recall
    metrics = pd.DataFrame({
        "Cancer F1": [cancer_f1],
        "System F1": [system_f1],
        "Cancer Precision": [cancer_precision],
        "Cancer Recall": [cancer_recall],
        "System Precision": [system_precision],
        "System Recall": [system_recall]
    })
    metrics.to_csv(Path(output_dir, "metrics.tsv"), index=False)
