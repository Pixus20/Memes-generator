import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import numpy as np

# Шлях до папки з фотографіями
data_dir = 'images'

# Задайте розмір зображень
img_size = (300, 273)
channels = 3 # 3 канали для кольорових фотографій

# Задайте розмір вхідного шуму (генератора)
noise_dim = 100

# Задайте генератор
generator = models.Sequential([
    layers.Dense(7 * 7 * 256, input_dim=noise_dim),
    layers.Reshape((7, 7, 256)),  # Reshape для створення 4D-виходу
    layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(channels, kernel_size=3, strides=1, padding='same', activation='sigmoid'),
])

# Задайте дискримінатор
discriminator = models.Sequential([
    layers.Conv2D(64, kernel_size=3, strides=2, input_shape=(img_size[0], img_size[1], channels), padding='same'),
    layers.LeakyReLU(),
    layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Задайте GAN, який об'єднує генератор та дискримінатор
discriminator.trainable = False
gan = models.Sequential([
    generator,
    discriminator
])

# Компіляція моделей
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Вивід структури генератора та дискримінатора
generator.summary()
discriminator.summary()

# Функція для генерації випадкового шуму
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

# Функція для тренування GAN
def train_gan(generator, discriminator, gan, data_generator, epochs=100, batch_size=64):
    batch_count = data_generator.samples // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):

            # Навчання дискримінатора
            noise = generate_noise(batch_size, noise_dim)
            generated_images = generator.predict(noise)
            real_images = data_generator.next()

            # Змінено: додано шар `Reshape` перед шаром `Dense`
            generated_images = tf.reshape(generated_images, (batch_size, img_size[0], img_size[1], channels))

            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))