

# Для начала, нам нужно подготовить набор данных для обучения нейронной сети. 
# Для этого мы можем использовать открытый набор данных MNIST, который содержит рукописные цифры от 0 до 9 в виде изображений размером 28x28 пикселей.

# Затем мы можем использовать библиотеку TensorFlow для создания и обучения нейронной сети. 
# Мы можем начать с простой архитектуры с несколькими полносвязными слоями, которые преобразуют пиксели изображения в векторы признаков, 
# а затем классифицируют эти векторы в соответствующие цифры.

# Когда модель будет обучена, мы можем использовать ее для распознавания рукописных цифр, 
#которые пользователь может нарисовать на экране. Мы можем использовать библиотеку Tkinter для создания простого пользовательского интерфейса,
# где пользователь может рисовать цифры и получать результаты распознавания.

import tensorflow as tf

# Загружаем данные MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Масштабируем данные до диапазона от 0 до 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Создаем модель нейронной сети
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Компилируем модель
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Оцениваем точность модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Используем модель для распознавания цифр
import numpy as np
from PIL import Image

# Загружаем изображение и преобразуем его в массив NumPy
img = Image.open('digit.png').convert('L')
img = img.resize((28, 28))
img_arr = np.array(img)

# Масштабируем значения пикселей до диапазона от 0 до 1 и инвертируем цвета
img_arr = 1 - img_arr / 255.0

# Распознаем цифру на изображении
prediction = model
