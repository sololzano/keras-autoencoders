from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, LeakyReLU, Conv1DTranspose, 
    Dense, Flatten, Activation, Reshape, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf
import numpy as np 
import os

class CAE(Model):
    def __init__(self, inp_shape, latent_dim, filters, act=None):
        super(CAE, self).__init__()
        
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
        self.rec_loss_tracker = tf.keras.metrics.Mean(name='rec_loss')

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

        # Latent variables
        self.latent = Sequential([
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
            self.rec_loss_tracker
        ]

    def encode(self, x):
        features = self.encoder(x)
        z = self.latent(features)

        return z

    def decode(self, z):
        z = self.reshape(z)
        x_hat = self.decoder(z)
        x_hat = self.out(x_hat)

        return x_hat

    def call(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_hat = self(x)

            rec_loss = tf.keras.losses.mean_squared_error(
                x, x_hat
            )
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.rec_loss_tracker.update_state(rec_loss)

        return {
            'rec_loss': self.rec_loss_tracker.result()
        }