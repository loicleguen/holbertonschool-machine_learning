#!/usr/bin/env python3
"""
Module defining the WGAN_clip class inheriting from tf.keras.Model.
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
    Class implementing a Wasserstein GAN with weight clipping.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes the WGAN_clip instance.

        Args:
            generator: Keras model for the generator network.
            discriminator: Keras model for the discriminator network.
            latent_generator: Function generating latent vectors.
            real_examples: Tensor containing real dataset samples.
            batch_size: Integer, batch size for training steps.
            disc_iter: Integer, number of discriminator training iterations
                       per step.
            learning_rate: Float, learning rate for Adam optimizers.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Define the generator loss and optimizer:
        # loss = -E[D(G(z))]
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        # Define the discriminator loss and optimizer:
        # loss = E[D(G(z))] - E[D(x)]
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(x) - tf.math.reduce_mean(y))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a fake sample using the generator model.

        Args:
            size: Optional integer specifying the
                    number of samples to generate.
            training: Boolean indicating whether the call is during training.

        Returns:
            Tensor of fake samples produced by the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a random sample of real data.

        Args:
            size: Optional integer specifying the number of real samples.

        Returns:
            Tensor of real samples randomly picked from real_examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Performs one training step for both the discriminator and generator.

        Args:
            useless_argument: Dummy argument required by Keras fit API.

        Returns:
            Dictionary containing 'discr_loss' and 'gen_loss'.
        """
        # 1. Train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                real_output = self.discriminator(
                    real_sample, training=True)
                fake_output = self.discriminator(
                    fake_sample, training=True)

                discr_loss = self.discriminator.loss(
                    fake_output, real_output)

            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

            # Clip discriminator weights between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # 2. Train the generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(
                fake_sample, training=True)
            gen_loss = self.generator.loss(fake_output)

        grads = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
