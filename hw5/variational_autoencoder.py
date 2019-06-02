'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import argparse

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="Load VAE model's trained weights")
    parser.add_argument("-f", "--fix", action="store_true",
                        help="Fix the variance vector to constant ones during training")
    parser.add_argument("-c", "--conv", action="store_true",
                        help="Train the VAE model with a ConvNet architecture")
    args = parser.parse_args()

    if args.conv:
        suffix = "_conv"
        intermediate_dim = 16
        input_shape = (28, 28, 1)

        x = Input(shape=input_shape, name='encoder_input')
        h = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        h = MaxPooling2D((2, 2), padding='same')(h)
        h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)

        # shape info needed to build decoder model
        shape = K.int_shape(h)

        # generate latent vector Q(z|X)
        h = Flatten()(h)
        h = Dense(intermediate_dim, activation='relu')(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        h_decoded = Dense(intermediate_dim, activation='relu')(z)
        h_decoded = Dense(shape[1] * shape[2] * shape[3], activation='relu')(h_decoded)
        h_decoded = Reshape((shape[1], shape[2], shape[3]))(h_decoded)
        h_decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        h_decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(h_decoded)
        h_decoded = UpSampling2D((2, 2))(h_decoded)
        x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h_decoded)

        # instantiate VAE model
        vae = Model(x, x_decoded_mean, name='vae')

        # instantiate generator model
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = Dense(intermediate_dim, activation='relu')(decoder_input)
        _h_decoded = Dense(shape[1] * shape[2] * shape[3], activation='relu')(_h_decoded)
        _h_decoded = Reshape((shape[1], shape[2], shape[3]))(_h_decoded)
        _h_decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(_h_decoded)
        _h_decoded = UpSampling2D((2, 2))(_h_decoded)
        _h_decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(_h_decoded)
        _h_decoded = UpSampling2D((2, 2))(_h_decoded)
        _x_decoded_mean = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(_h_decoded)
        generator = Model(inputs=decoder_input, outputs=_x_decoded_mean)

    else:
        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)

        if args.fix:
            z_log_var = Input(tensor=K.zeros(latent_dim))
            suffix = "_fixed_var"
        else:
            z_log_var = Dense(latent_dim)(h)
            suffix = ""

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # instantiate VAE model
        if args.fix:
            vae = Model([x, z_log_var], x_decoded_mean)
        else:
            vae = Model(x, x_decoded_mean)

        # instantiate generator model
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(inputs=decoder_input, outputs=_x_decoded_mean)

    # instantiate encoder model
    encoder = Model(x, z_mean)

    # compute VAE loss
    if args.conv:
        xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(x), K.flatten(x_decoded_mean))
    else:
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    if args.conv:
        x_train = x_train.reshape((len(x_train), x_train.shape[1], x_train.shape[2], 1))
        x_test = x_test.reshape((len(x_test), x_train.shape[1], x_train.shape[2], 1))
    else:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(x_train.shape)
    print(x_test.shape)
    data = (x_test, y_test)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the VAE
        vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))

        vae.save_weights("vae_mnist" + suffix + ".h5")

    digit_size = 28

    # section (c)
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    fig = plt.figure()
    for digit in range(10):
        i = np.where(y_test == digit)[0][0]
        ax = fig.add_subplot(2, 5, digit + 1)
        plt.axis('off')
        ax.imshow(x_test[i].reshape(digit_size, digit_size))
        ax.set_title("(%.2f, %.2f)" % tuple(x_test_encoded[i]))
    plt.tight_layout()
    plt.savefig("results/hw5_latent_space" + suffix + ".png")

    # section (d)
    z_sample = np.array([[0.5, 0.2]]) * epsilon_std
    x_decoded = generator.predict(z_sample)
    plt.imsave("results/hw5_generated_digit" + suffix + ".png",
               x_decoded[0].reshape(digit_size, digit_size))

    # section (e)
    i0 = np.where(y_test == 0)[0][0]
    i8 = np.where(y_test == 8)[0][0]
    rep0 = x_test_encoded[i0]
    rep8 = x_test_encoded[i8]
    ps = np.random.random_sample(10)
    points = np.array([rep0, rep8] +
                      [p*rep0 + (1-p)*rep8 for p in ps])  # points from the line connecting the two
    fig = plt.figure()
    for i, point in enumerate(sorted(points.tolist())):
        z_sample = np.array([point]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        ax = fig.add_subplot(1, 12, i+1)
        plt.axis('off')
        ax.imshow(x_decoded[0].reshape(digit_size, digit_size))
    plt.savefig("results/hw5_interpolation" + suffix + ".png")
