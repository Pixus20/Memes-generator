import tensorflow as tf 
from keras import layers, models
input_size = 50
model = models.Sequential()
# Додавання шарів до моделі
model.add(layers.Dense(128, activation='relu', input_shape=(input_size,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вивід структури моделі
model.summary()