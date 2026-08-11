#!/usr/bin/env python3
"""
Module defining the WGAN_GP class inheriting from tf.keras.Model.
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    Class implementing a Wasserstein GAN with Gradient Penalty (WGAN-GP).
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN_GP instance.

        Args:
            generator: Keras model for the generator network.
            discriminator: Keras model for the discriminator network.
            latent_generator: Function generating latent vectors.
            real_examples: Tensor containing real dataset samples.
            batch_size: Integer, batch size for training steps.
            disc_iter: Integer, number of discriminator training iterations
                       per step.
            learning_rate: Float, learning rate for Adam optimizers.
            lambda_gp: Float, coefficient for the gradient penalty.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates interpolated samples between real and fake samples.

        Args:
            real_sample: Tensor of real samples.
            fake_sample: Tensor of fake samples.

        Returns:
            Tensor of interpolated samples: u * real + (1 - u) * fake.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the discriminator on interpolated
        samples.

        Args:
            interpolated_sample: Tensor of interpolated samples.

        Returns:
            Tensor representing the gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Performs one training step for both the discriminator and generator.

        Args:
            useless_argument: Dummy argument required by Keras fit API.

        Returns:
            Dictionary containing 'discr_loss', 'gen_loss', and 'gp'.
        """
        # 1. Train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                interpolated_sample = self.get_interpolated_sample(
                    real_sample, fake_sample)

                real_output = self.discriminator(
                    real_sample, training=True)
                fake_output = self.discriminator(
                    fake_sample, training=True)

                discr_loss = self.discriminator.loss(
                    fake_output, real_output)

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

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

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replaces generator and discriminator weights with pre-trained weights
        loaded from .h5 files.

        Args:
            gen_h5: Path to the generator's .h5 weight file.
            disc_h5: Path to the discriminator's .h5 weight file.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
