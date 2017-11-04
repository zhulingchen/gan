# import general packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from os import path
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
import h5py
import random
from tqdm import tqdm
from IPython import display

# import keras packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.utils import plot_model  # for model visualization, need to install Graphviz (.msi for Windows), pydot (pip install), graphviz (pip install) and set PATH for Graphviz
from keras.optimizers import *
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

n_class = 10
n_cont = 2
method = 'unsupervised'

def data_normalization(X):
    return (X / 127.5) - 1

def data_denormalization(X):
    return (X + 1.) / 2.

def closest_node(pts, centers):
    return np.argmin(cdist(pts, centers, 'sqeuclidean'), axis=1)

# freeze weights in the discriminator for stacked training
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def gaussian_loss(y_true, y_pred):
    mean = y_pred[:, 0, :]
    log_stddev = y_pred[:, 1, :]
    y_true = y_true[:, 0, :]
    normalized = (y_true - mean) / (K.exp(log_stddev) + K.epsilon())
    loss = (log_stddev + 0.5 * K.square(normalized))
    loss = K.mean(loss)
    return loss

# plot loss function
def plot_loss(losses, save=False, saveFileName=None, method='dcgan'):
    plt.figure(figsize=(10, 10))
    d_loss = np.array(losses['d'])
    g_loss = np.array(losses['g'])
    if method is 'dcgan':
        plt.loglog(d_loss, label='discriminitive loss')
        plt.loglog(g_loss, label='generative loss')
    elif method is 'infogan':
        plt.loglog(d_loss[:, 0], label='discriminitive loss (total)')
        plt.loglog(d_loss[:, 1], label='discriminitive loss (binary)')
        plt.loglog(d_loss[:, 2], label='discriminitive loss (categorical)')
        plt.loglog(d_loss[:, 3], label='discriminitive loss (continuous)')
        plt.loglog(g_loss[:, 0], label='generative loss (total)')
        plt.loglog(g_loss[:, 1], label='generative loss (binary)')
        plt.loglog(g_loss[:, 2], label='generative loss (categorical)')
        plt.loglog(g_loss[:, 3], label='generative loss (continuous)')
    plt.legend()
    if save:
        if saveFileName is not None:
            plt.savefig(saveFileName)
    else:
        plt.show()
    plt.clf()
    plt.close()

def get_image_batch(X, y, batch_size=32):
    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx], y[idx]

def get_noise_sample(dimvec, batch_size=32, method='dcgan'):
    noise = np.random.uniform(-1, 1, size=(batch_size, dimvec))
    if method is 'dcgan':
        return noise
    elif method is 'infogan':
        label = np.random.randint(0, n_class, size=(batch_size, 1))
        label = np_utils.to_categorical(label, num_classes=n_class)
        cont_2d = np.random.uniform(-1, 1, size=(batch_size, n_cont))
        cont_3d = np.expand_dims(cont_2d, axis=1)
        cont_3d = np.repeat(cont_3d, 2, axis=1)
        return noise, label, cont_2d, cont_3d

# plot generated images
def plot_gen(generator, n_example=16, dim=(4, 4), figsize=(10, 10), save=False, saveFileName=None, method='dcgan'):
    dimvec = generator.layers[0].input_shape[1]
    if method is 'dcgan':
        noise = get_noise_sample(dimvec, batch_size=n_example, method='dcgan')
        image_gen = generator.predict(noise)
        image_gen = data_denormalization(image_gen)
        plt.figure(figsize=figsize)
        for i in range(image_gen.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(image_gen[i, 0, :, :], cmap='gray')
            plt.axis('off')
    elif method is 'infogan_cat':
        n_image_col = dim[1]
        plt.figure(figsize=figsize)
        for i in range(n_class):
            noise, _, _, _ = get_noise_sample(dimvec, batch_size=n_image_col, method='infogan')
            label = np.repeat(i, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=n_class)
            cont = np.repeat(np.zeros((1, n_cont)), n_image_col, axis=0)  # (np.expand_dims(np.linspace(-1, 1, num=n_image_col), axis=1), n_cont, axis=1)
            image_gen = generator.predict([noise, label, cont])
            image_gen = data_denormalization(image_gen)
            for j in range(n_image_col):
                plt.subplot(n_class, n_image_col, i*n_image_col+j+1)
                plt.imshow(image_gen[j, 0, :, :], cmap='gray')
                plt.axis('off')
    elif method is 'infogan_cont':
        n_image_row, n_image_col = dim[0], dim[1]
        plt.figure(figsize=figsize)
        label_number = 0
        cont_range_row = np.linspace(-1, 1, num=n_image_row)
        cont_range_col = np.linspace(-1, 1, num=n_image_col)
        for i in range(n_image_row):
            noise, _, _, _ = get_noise_sample(dimvec, batch_size=n_image_col, method='infogan')
            label = np.repeat(label_number, n_image_col).reshape(-1, 1)
            label = np_utils.to_categorical(label, num_classes=n_class)
            cont = np.concatenate([np.array([cont_range_row[i], cont_range_col[j]]).reshape(1, -1) for j in range(n_image_col)])
            image_gen = generator.predict([noise, label, cont])
            image_gen = data_denormalization(image_gen)
            for j in range(n_image_col):
                plt.subplot(n_image_row, n_image_col, i*n_image_col+j+1)
                plt.imshow(image_gen[j, 0, :, :], cmap='gray')
                plt.axis('off')
    if save:
        if saveFileName is not None:
            plt.savefig(saveFileName)
    else:
        plt.tight_layout()
        plt.show()
    plt.clf()
    plt.close()

# the data, shuffled and split between train and test sets
def load_preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_row = X_train.shape[1]
    n_col = X_train.shape[2]
    X_train = np.reshape(X_train, (n_train, 1, n_row, n_col)).astype('float32')
    X_test = np.reshape(X_test, (n_test, 1, n_row, n_col)).astype('float32')
    X_train = data_normalization(X_train)
    X_test = data_normalization(X_test)
    return (X_train, y_train), (X_test, y_test)

# build generative model for DCGAN
def build_generator_for_dcgan(n_row, n_col, n_ch=128, dim_noise=100):
    g_opt = Adam(lr=1E-4, beta_1=0.5)
    g_in = Input(shape=(dim_noise,))
    x = Dense(1024)(g_in)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dense(n_ch * (n_row/4) * (n_col/4))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Reshape((n_ch, n_row/4, n_col/4))(x)
    x = Conv2DTranspose(n_ch/2, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='tanh')(x)
    generator = Model(g_in, g_out)
    generator.compile(optimizer=g_opt, loss='mse')
    print 'Summary of Generator (for DCGAN)'
    generator.summary()
    return generator

# build generative model for InfoGAN
def build_generator_for_infogan(n_row, n_col, n_ch=128, dim_noise=100, dim_cat=0, dim_cont=0):
    g_opt = Adam(lr=1E-4, beta_1=0.5)
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(dim_cat,))
    g_in_cont = Input(shape=(dim_cont,))
    g_in = concatenate([g_in_noise, g_in_cat, g_in_cont])
    x = Dense(1024)(g_in)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Dense(n_ch * (n_row/4) * (n_col/4))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Reshape((n_ch, n_row/4, n_col/4))(x)
    x = Conv2DTranspose(n_ch/2, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='tanh')(x)
    generator = Model([g_in_noise, g_in_cat, g_in_cont], g_out)
    generator.compile(optimizer=g_opt, loss='mse')
    print 'Summary of Generator (for InfoGAN)'
    generator.summary()
    return generator

# build discriminator model (base)
def build_discriminator_base(n_row, n_col, n_ch=128, drRate=0.25, leaky_relu_alpha=0.2):
    d_in = Input(shape=(1, n_row, n_col))
    x = Conv2D(n_ch/2, (3, 3), strides=2, padding='same')(d_in)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_ch, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization(axis=1)(x)
    d_base_out = LeakyReLU(alpha=leaky_relu_alpha, name='d_penultimate')(x)
    return d_in, d_base_out

# build discriminator model for DCGAN
def build_discriminator_for_dcgan(n_row, n_col, n_ch=128, drRate=0.25, leaky_relu_alpha=0.2):
    d_opt = Adam(lr=1E-4, beta_1=0.5)
    d_in, d_base_out = build_discriminator_base(n_row, n_col, n_ch, drRate, leaky_relu_alpha)
    d_out = Dense(2, activation='softmax')(d_base_out)
    discriminator = Model(d_in, d_out)
    discriminator.compile(optimizer=d_opt, loss='binary_crossentropy')
    print 'Summary of Discriminator (for DCGAN)'
    discriminator.summary()
    return discriminator

# build discriminator model for InfoGAN
losses_infogan = ['binary_crossentropy', 'categorical_crossentropy', gaussian_loss]
list_weights = [1, 1, 1]
def build_discriminator_for_infogan(n_row, n_col, n_ch=128, drRate=0.25, leaky_relu_alpha=0.2):
    d_opt = Adam(lr=1E-4, beta_1=0.5)
    d_in, d_base_out = build_discriminator_base(n_row, n_col, n_ch, drRate, leaky_relu_alpha)
    d_out_validity = Dense(2, activation='softmax')(d_base_out)  # validity output

    # categorical output
    label_model = Dense(128)(d_base_out)
    label_model = BatchNormalization(axis=1)(label_model)
    label_model = LeakyReLU(alpha=leaky_relu_alpha)(label_model)
    label = Dense(n_class, activation='softmax')(label_model)

    def linmax(x):
        return K.maximum(x, -16)
    def linmax_shape(input_shape):
        return input_shape

    # continuous output
    mean = Dense(n_cont, activation='linear')(label_model)  # means of n_cont (2) latent continuous variables
    log_stddev = Dense(n_cont)(label_model)
    log_stddev = Lambda(linmax, output_shape=linmax_shape)(log_stddev)  # logarithmic standard deviation of n_cont (2) latent continuous variables
    mean = Reshape((1, n_cont))(mean)
    log_stddev = Reshape((1, n_cont))(log_stddev)
    cont = concatenate([mean, log_stddev], axis=1)

    discriminator = Model(d_in, [d_out_validity, label, cont])
    discriminator.compile(optimizer=d_opt, loss=losses_infogan, loss_weights=list_weights)
    print 'Summary of Discriminator (for InfoGAN)'
    discriminator.summary()
    return discriminator

# build stacked DCGAN model
def build_dcgan(generator, discriminator):
    gan_opt = Adam(lr=1E-4, beta_1=0.5)
    gan_in = Input(shape=generator.layers[0].input_shape[1:])
    gen_out = generator(gan_in)
    set_trainable(discriminator, False)
    gan_out = discriminator(gen_out)
    gan = Model(gan_in, gan_out)
    gan.compile(optimizer=gan_opt, loss='binary_crossentropy')
    set_trainable(discriminator, True)
    print 'Summary of the DCGAN model'
    gan.summary()
    return gan

# build stacked InfoGAN model
def build_infogan(generator, discriminator):
    gan_opt = Adam(lr=1E-4, beta_1=0.5)
    gan_in_noise = Input(shape=generator.layers[0].input_shape[1:])
    gan_in_cat = Input(shape=generator.layers[1].input_shape[1:])
    gan_in_cont = Input(shape=generator.layers[2].input_shape[1:])
    gen_out = generator([gan_in_noise, gan_in_cat, gan_in_cont])
    set_trainable(discriminator, False)
    d_out, target_label, target_cont = discriminator(gen_out)
    gan = Model([gan_in_noise, gan_in_cat, gan_in_cont], [d_out, target_label, target_cont])
    gan.compile(optimizer=gan_opt, loss=losses_infogan, loss_weights=list_weights)
    set_trainable(discriminator, True)
    print 'Summary of the InfoGAN model'
    gan.summary()
    return gan

# GAN train function
def train_for_n(generator, discriminator, gan, n_train, epochs=100, plt_frq=25, batch_size=32):
    for e in range(epochs):
        print 'epoch: %d' % (e + 1)
        nb = np.floor(n_train/batch_size)
        batch_counter = 0
        progbar = generic_utils.Progbar(nb*batch_size)
        for image_batch, label_batch in get_image_batch(X_train, y_train, batch_size):
            # train discriminator
            if isinstance(gan.layers[-1].output_shape, list):  # infogan
                # generate fake images, labels, and mean/log_stddev
                noise_gen, label_gen, cont_gen_2d, cont_gen_3d = get_noise_sample(generator.layers[0].input_shape[1], batch_size=batch_size, method='infogan')
                image_gen = generator.predict([noise_gen, label_gen, cont_gen_2d])
                # train discriminator
                if batch_counter % 2 == 0:
                    X = image_batch
                    y = np.ones((batch_size,))
                    if method is 'unsupervised':
                        label = label_gen
                    elif method is 'supervised':
                        label = label_batch
                        label = np_utils.to_categorical(label, num_classes=n_class)
                else:
                    X = image_gen
                    y = np.zeros((batch_size,))
                    label = label_gen
                y = np_utils.to_categorical(y, num_classes=2)
                d_loss = discriminator.train_on_batch(X, [y, label, cont_gen_3d])
            else:  # dcgan
                # generate fake images
                noise_gen = get_noise_sample(generator.layers[0].input_shape[1], batch_size=batch_size, method='dcgan')
                image_gen = generator.predict(noise_gen)
                # train discriminator
                if batch_counter % 2 == 0:
                    X = image_batch
                    y = np.ones((batch_size,))
                else:
                    X = image_gen
                    y = np.zeros((batch_size,))
                y = np_utils.to_categorical(y, num_classes=2)
                d_loss = discriminator.train_on_batch(X, y)
            losses['d'].append(d_loss)

            # train generator-discriminator stack on input noise to non-generated output class
            if isinstance(gan.layers[-1].output_shape, list):  # infogan
                noise_train, label_train, cont_train_2d, cont_train_3d = get_noise_sample(generator.layers[0].input_shape[1], batch_size=batch_size, method='infogan')
                y_1 = np.ones((batch_size,))
                y_1 = np_utils.to_categorical(y_1, num_classes=2)
                set_trainable(discriminator, False)
                g_loss = gan.train_on_batch([noise_train, label_train, cont_train_2d],
                                               [y_1, label_train, cont_train_3d])
            else:  # dcgan
                noise_train = get_noise_sample(generator.layers[0].input_shape[1], batch_size=batch_size, method='dcgan')
                y_1 = np.ones((batch_size,))
                y_1 = np_utils.to_categorical(y_1, num_classes=2)
                set_trainable(discriminator, False)
                g_loss = gan.train_on_batch(noise_train, y_1)
            set_trainable(discriminator, True)
            losses['g'].append(g_loss)

            # update progress bar
            if isinstance(gan.layers[-1].output_shape, list):  # infogan
                progbar.add(batch_size, values=[("D total", d_loss[0]), ("D bin", d_loss[1]), ("D cat", d_loss[2]), ("D cont", d_loss[3]),
                                                ("G total", g_loss[0]), ("G bin", g_loss[1]), ("G cat", g_loss[2]), ("G cont", g_loss[3])])
            else:
                progbar.add(batch_size, values=[("D total", d_loss), ("G total", g_loss)])

            # updates plots
            if ((batch_counter + 1) % plt_frq == 0) or (batch_counter == nb-1):
                if isinstance(gan.layers[-1].output_shape, list):  # infogan
                    loss_saveFileName = './infogan_mnist_cat_epoch%d_batch%d_loss.pdf' % (e + 1, batch_counter + 1)
                    plot_loss(losses, True, loss_saveFileName, method='infogan')
                    image_gen_saveFileName = './infogan_mnist_cat_epoch%d_batch%d.pdf' % (e + 1, batch_counter + 1)
                    plot_gen(generator, 100, (10, 10), (15, 15), True, image_gen_saveFileName, method='infogan_cat')
                    image_gen_saveFileName = './infogan_mnist_cont_epoch%d_batch%d.pdf' % (e + 1, batch_counter + 1)
                    plot_gen(generator, 100, (10, 10), (15, 15), True, image_gen_saveFileName, method='infogan_cont')
                    print 'epoch %d / batch %d: Generative Loss: %.4f, Discriminative Loss: %.4f' % (e + 1, batch_counter + 1, g_loss[0], d_loss[0])
                else:  # dcgan
                    loss_saveFileName = './dcgan_mnist_epoch%d_batch%d_loss.pdf' % (e + 1, batch_counter + 1)
                    plot_loss(losses, True, loss_saveFileName, method='dcgan')
                    image_gen_saveFileName = './dcgan_mnist_epoch%d_batch%d.pdf' % (e + 1, batch_counter + 1)
                    plot_gen(generator, 100, (10, 10), (15, 15), True, image_gen_saveFileName, method='dcgan')
                    print 'epoch %d / batch %d: Generative Loss: %.4f, Discriminative Loss: %.4f' % (e + 1, batch_counter + 1, g_loss, d_loss)

            batch_counter += 1
            if batch_counter == nb:
                break

# K.set_value(g_opt.lr, 1e-6)
# K.set_value(d_opt.lr, 1e-5)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_preprocess_mnist()
    n_train = X_train.shape[0]
    n_row, n_col = X_train.shape[2:]
    n_epoch = 50

    # train InfoGAN
    losses = {'d': [], 'g': []}
    generator_infogan = build_generator_for_infogan(n_row, n_col, dim_noise=64, dim_cat=n_class, dim_cont=n_cont)
    plot_model(generator_infogan, to_file='./generator_infogan_model.pdf', show_shapes=True)
    discriminator_infogan = build_discriminator_for_infogan(n_row, n_col)
    plot_model(discriminator_infogan, to_file='./discriminator_infogan_model.pdf', show_shapes=True)
    infogan = build_infogan(generator_infogan, discriminator_infogan)
    plot_model(infogan, to_file='./infogan_model.pdf', show_shapes=True)
    generator_weights_file = './generator_infogan_weights_epoch%d_%s.hdf' % (n_epoch, method)
    discriminator_weights_file = './discriminator_infogan_weights_epoch%d_%s.hdf' % (n_epoch, method)
    gan_weights_file = './infogan_weights_epoch%d_%s.hdf' % (n_epoch, method)
    if path.isfile(generator_weights_file) and path.isfile(discriminator_weights_file) and path.isfile(gan_weights_file):
        generator_infogan.load_weights(generator_weights_file)
        discriminator_infogan.load_weights(discriminator_weights_file)
        infogan.load_weights(gan_weights_file)
    else:
        train_for_n(generator_infogan, discriminator_infogan, infogan, n_train, epochs=n_epoch, plt_frq=2000, batch_size=32)
        generator_infogan.save_weights(generator_weights_file)
        discriminator_infogan.save_weights(discriminator_weights_file)
        infogan.save_weights(gan_weights_file)

    # train DCGAN
    losses = {'d': [], 'g': []}
    generator_dcgan = build_generator_for_dcgan(n_row, n_col, dim_noise=64)
    plot_model(generator_dcgan, to_file='./generator_dcgan_model.pdf', show_shapes=True)
    discriminator_dcgan = build_discriminator_for_dcgan(n_row, n_col)
    plot_model(discriminator_dcgan, to_file='./discriminator_dcgan_model.pdf', show_shapes=True)
    dcgan = build_dcgan(generator_dcgan, discriminator_dcgan)
    plot_model(dcgan, to_file='./dcgan_model.pdf', show_shapes=True)
    generator_weights_file = './generator_dcgan_weights_epoch%d.hdf' % n_epoch
    discriminator_weights_file = './discriminator_dcgan_weights_epoch%d.hdf' % n_epoch
    gan_weights_file = './dcgan_weights_epoch%d.hdf' % n_epoch
    if path.isfile(generator_weights_file) and path.isfile(discriminator_weights_file) and path.isfile(gan_weights_file):
        generator_dcgan.load_weights(generator_weights_file)
        discriminator_dcgan.load_weights(discriminator_weights_file)
        dcgan.load_weights(gan_weights_file)
    else:
        train_for_n(generator_dcgan, discriminator_dcgan, dcgan, n_train, epochs=n_epoch, plt_frq=2000, batch_size=32)
        generator_dcgan.save_weights(generator_weights_file)
        discriminator_dcgan.save_weights(discriminator_weights_file)
        dcgan.save_weights(gan_weights_file)