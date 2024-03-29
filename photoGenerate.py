import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))

data_dir = 'images/cats'
img_size = (56,56)
channels = 3

noise_dim = 100
batch_size = 64
noise = generate_noise(batch_size, noise_dim)

image_generator = ImageDataGenerator(rescale=1./255)
data_generator = image_generator.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None
)

# Генератор model
generator = models.Sequential([
    layers.Dense(7 * 7 * 256, input_dim=noise_dim),
    layers.Reshape((7, 7, 256)),
    layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(channels, kernel_size=3, strides=2, padding='same', activation='tanh')
])
for images_batch in data_generator:
    print(f"Зображення отримані з генератора: {images_batch.shape}")
    break

# Дискримінатор model
discriminator = models.Sequential([
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(56, 56, channels)),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Загальна модель GAN
discriminator.trainable = False

gan_input = layers.Input(shape=(noise_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Цикл тренування
def train_gan(generator, discriminator, gan, data_generator, epochs=100, batch_size=64):
    batch_count = len(data_generator)

    for epoch in range(epochs):
        for _ in range(batch_count):
            noise = generate_noise(batch_size, noise_dim)
            generated_images = generator.predict(noise)
            real_images = data_generator.next()
           
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)
            
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, labels_real)            
          
            print(f"Епоха {epoch + 1}, Партія {_ + 1}/{batch_count}, Втрата D для реальних: {d_loss_real[0]}, Втрата D для сгенерованих: {d_loss_fake[0]}, Втрата G: {g_loss[0]}")
            save_generated_images(generated_images, epoch)

def save_generated_images(generated_images, epoch, rows=4, columns=4):
    count = 0  
    fig, axs = plt.subplots(rows, columns)
    for i in range(rows):
        for j in range(columns):
            axs[i, j].imshow(generated_images[count])
            axs[i, j].axis('off')
            count += 1
    plt.savefig(f"generated_images_epoch_{epoch}.png")
    plt.close()

train_gan(generator, discriminator, gan, data_generator, epochs=100, batch_size=64)