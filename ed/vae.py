from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, LeakyReLU, Conv1DTranspose, 
    Dense, Flatten, Activation, Reshape, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf
import numpy as np 
import os

class VAE_1D(Model):
    def __init__(self, inp_shape, latent_dim, filters, act=None):
        super(VAE_1D, self).__init__()

        assert inp_shape[0] % (2**len(filters)) == 0, 'Dimensions do not match'
        assert len(inp_shape) == 2, 'Input dimensions must be rank 2'
        assert latent_dim < inp_shape[0], 'Latent dimension cannot be greater than \
                                        input dimension'

        self.inp_shape = inp_shape
        self.latent_dim = latent_dim
        self.filters = filters
        self.last_dim = inp_shape[0] // (2**len(filters))
        self.act = act

        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.rec_loss_tracker = tf.keras.metrics.Mean(name='rec_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

        # Encoder
        self.encoder = Sequential([Input(inp_shape)])
        for f in filters:
            self.encoder.add(
                Sequential([
                    Conv1D(f, kernel_size=3, strides=2, padding='same'),
                    BatchNormalization(),
                    LeakyReLU()
                ])
            )

        # Latent space
        self.fc_mu = Sequential([
            Flatten(),
            Dense(latent_dim)
        ])
        self.fc_log_var = Sequential([
            Flatten(),
            Dense(latent_dim)
        ])

        # Reshape
        self.reshape = Sequential([
            Dense(self.last_dim * inp_shape[-1]),
            Reshape((self.last_dim, inp_shape[-1]))
        ])

        # Decoder
        n_filters = filters[::-1]
        self.decoder = Sequential()
        for f in n_filters:
            self.decoder.add(
                Sequential([
                    Conv1DTranspose(f, kernel_size=3, strides=2, 
                        padding='same', output_padding=1),
                    LeakyReLU(),
                ])
            )
        self.out = Sequential([
            Conv1D(inp_shape[-1], kernel_size=1, padding='same'),
            Activation(act) if act else Lambda(lambda x: x)
        ])

        self.build([None] + list(inp_shape))

    @property 
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.rec_loss_tracker, 
            self.kl_loss_tracker
        ]

    def encode(self, x):
        features = self.encoder(x)
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)

        return [mu, log_var]

    @tf.function
    def reparameterize(self, mu, log_var):
        std = tf.exp(.5 * log_var)
        eps = tf.keras.backend.random_normal(shape=tf.shape(mu))

        return mu + std * eps

    def decode(self, z):
        z = self.reshape(z)
        x_hat = self.decoder(z)
        x_hat = self.out(x_hat)

        return x_hat

    def call(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)

        return [x_hat, mu, log_var]

    @tf.function 
    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_hat, mu, log_var = self(x)

            # Reconstruction error
            rec_loss = tf.keras.losses.mean_squared_error(x, x_hat)
            rec_loss = tf.reduce_mean(rec_loss)

            # KL divergence
            kl_weight = tf.cast(1 / tf.shape(x)[0], dtype=tf.float32)
            kl_loss = -.5 * tf.reduce_sum(1 + log_var - mu**2 - tf.exp(log_var), axis=-1)
            kl_loss = kl_weight * tf.reduce_mean(kl_loss, axis=0)

            total_loss = rec_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'rec_loss': self.rec_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }