from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import DataPipe
import numpy as np
import random

img_rows = 100
img_cols = 170
img_shape = (img_rows, img_cols, 3)


def build_generator():
    """
    :return: the generator model.
    """
    noise_shape = (100,)
    model = Sequential(name="generator")
    model.add(Dense(6 * 11 * 512, input_shape=noise_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((6, 11, 512)))

    model.add(Conv2DTranspose(256, 3, 2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, 3, 2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, 3, 2))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, 4, 2, activation='tanh'))
    model.add(BatchNormalization(momentum=0.9))

    model.summary()
    noise = Input(shape=noise_shape)
    image = model(noise)
    return Model(noise, image)


def build_discriminator():
    """
    :return: the discriminator model.
    """
    model = Sequential(name="discriminator")

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    image = Input(shape=img_shape)
    validity = model(image)
    return Model(image, validity)


def train(epochs, batch_size=128, save_interval=50):
    """
    :param epochs:the number of epochs
    :type epochs: int
    :param batch_size:the batch
    :param save_interval:the number frequency which it will save the model and output images
    """
    try:
        generator = load_model('models/generator_model.h5')
        discriminator = load_model('models/discriminator_model.h5')
        combined = load_model('models/combined_model.h5')
        print("loading...")
    except OSError:
        print("creating new model...")
        optimizer = Adam(0.0002, 0.5)
        discriminator = build_discriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        generator = build_generator()
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        z = Input(shape=(100,))
        discriminator.trainable = False
        valid = discriminator(generator(z))
        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    half_batch = int(batch_size / 2)

    dp = DataPipe()
    dp.load_all_images()
    for epoch in range(epochs):
        images = dp.get_butch(half_batch)

        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_images = generator.predict(noise)

        labels = np.ones((half_batch, 1))
        d_loss_real = discriminator.train_on_batch(images, labels)
        d_loss_fake = discriminator.train_on_batch(gen_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise1 = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise1, valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_images(epoch, generator)
            generator.save('models/generator_model.h5')
            discriminator.save('models/discriminator_model.h5')
            combined.save('models/combined_model.h5')


def save_images(epoch, generator):
    """
    generate an car image with the generator and save it in the batches_output_images folder
    :param epoch:the number of epochs
    :type epoch: int
    :param generator:the generator model
    :type generator: Model
    """
    n = 4
    noise = np.random.normal(0, 1, (n * n, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(gen_images[i])
    filename = f"batches_output_images/generated_plot_epoch-{epoch}.png"
    plt.savefig(filename)
    plt.close()


def generate_car():
    """
    generate an car image with the generator and save it in the new_predictions folder
    """
    generator_ = load_model('models/generator_model.h5')
    noise = np.random.normal(0, 1, (1, 100))
    gen_image = generator_.predict(noise)
    gen_image = (gen_image + 1) * 127.5
    gen_image = gen_image.astype(int)
    plt.axis("off")
    plt.imshow(gen_image[0])
    plt.savefig(f"new_predictions/p{random.randint(0, 999)}.png")
    plt.close()


for _ in range(10):
    generate_car()


