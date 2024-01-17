import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Function to generate noise
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

# Data parameters
data_dir = 'images'
img_size = (300, 273)
channels = 3

# Noise dimensions
noise_dim = 100
batch_size = 64
noise = generate_noise(batch_size, noise_dim)

# Image generator for real images
image_generator = ImageDataGenerator(rescale=1./255)
data_generator = image_generator.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None
)

# Generator model
generator = models.Sequential([
    layers.Dense(7 * 7 * 256, input_dim=noise_dim),
    layers.Reshape((7, 7, 256)),
    layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(channels, kernel_size=3, strides=1, padding='same', activation='sigmoid')
])

# ...

# Цикл тренування
def train_gan(generator, discriminator, gan, data_generator, epochs=100, batch_size=64):
    batch_count = data_generator.samples // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            noise = generate_noise(batch_size, noise_dim)
            generated_images = generator.predict(noise)
            real_images = data_generator.next()

            # Мітки для тренування
            labels_real = np.ones((batch_size, 64))
            labels_fake = np.zeros((batch_size, 64))

            # Тренування дискримінатора
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            # Тренування генератора
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, labels_real)

            # Виведення втрат
            print(f"Партія {batch_count}, Втрата D для реальних: {d_loss_real[0]}, Втрата D для сгенерованих: {d_loss_fake[0]}, Втрата G: {g_loss[0]}")
            if epoch % 5 == 0:
                save_generated_images(generated_images, epoch)

def save_generated_images(images, epoch, rows=4, columns=4):
    fig, axs = plt.subplots(rows, columns)
    count = 0
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow(images[count])
            axs[i, j].axis('off')
            count += 1
    plt.savefig(f"generated_images_epoch_{epoch}.png")
    plt.show()