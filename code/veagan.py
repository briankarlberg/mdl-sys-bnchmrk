import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Assuming input dimensions and latent dimensions are defined

latent_dim = 10
cancer_column = "Cancer_type"
system_column = "System"

def build_encoder():
    inputs = tf.keras.Input(shape=(data.shape[1],))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')
    return encoder


def build_decoder():
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(data.shape[1], activation='sigmoid')(x)

    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    return decoder


def build_discriminator():
    inputs = tf.keras.Input(shape=(data.shape[1]))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    discriminator = tf.keras.Model(inputs, outputs, name='discriminator')
    return discriminator


class VAEGAN(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator):
        super(VAEGAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def compile(self, vae_optimizer, gan_optimizer, loss_fn, **kwargs):
        super(VAEGAN, self).compile()
        self.vae_optimizer = vae_optimizer
        self.gan_optimizer = gan_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data
        real_data = data

        # VAE part
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(real_data)
            z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=(latent_dim,))
            reconstructed_data = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(losses.binary_crossentropy(real_data, reconstructed_data))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            vae_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(vae_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.vae_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

        # GAN part
        with tf.GradientTape() as tape:
            fake_data = self.decoder(z)
            fake_logits = self.discriminator(fake_data)
            real_logits = self.discriminator(real_data)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_logits), fake_logits)
            d_loss_real = self.loss_fn(tf.ones_like(real_logits), real_logits)
            d_loss = (d_loss_real + d_loss_fake) / 2
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.gan_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        return {"vae_loss": vae_loss, "dis_loss": d_loss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAEGAN')
    parser.add_argument('--data', "-d", type=str, required=True, help='Path to the data')
    args = parser.parse_args()

    data = pd.read_csv(args.data, sep='\t')

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

    # split data into train / test
    # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_train_data = scaler.fit_transform(train_data)
    # scaled_test_data = scaler.transform(test_data)

    scaled_data = scaler.fit_transform(data)

    train_dataset = tf.data.Dataset.from_tensor_slices(scaled_data).batch(64)

    encoder = build_encoder()
    decoder = build_decoder()
    discriminator = build_discriminator()

    encoder.summary()
    decoder.summary()
    discriminator.summary()

    vaegan = VAEGAN(encoder, decoder, discriminator)
    vaegan.compile(vae_optimizer=optimizers.Adam(1e-4), gan_optimizer=optimizers.Adam(1e-4),
                   loss_fn=losses.BinaryCrossentropy(from_logits=True))

    history = vaegan.fit(train_dataset, epochs=50)

    # Correcting batch effects
    _, _, x_test_encoded = vaegan.encoder.predict(scaled_data)

    # Save the latent space
    x_test_encoded = pd.DataFrame(x_test_encoded)
    x_test_encoded.to_csv("latent_space.csv", index=False)
