import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import Sequential


reviews_train, reviews_validate, reviews_test = tfds.load(
    'imdb_reviews',
    split=('train[:80%]', 'train[80%:90%]', 'test'),
    as_supervised=True
)

for i in reviews_train.take(10):
    print('Review text: ', i[0].numpy())
    print('Review sentiment: ', i[1].numpy(), '\n')

#with tf hub we can feed our model directly as string
hub_layer = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1', input_shape=[], dtype=tf.string, trainable=True)

model = keras.Sequential([
  hub_layer,
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dropout(0.24),
  keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    reviews_train.shuffle(10000).batch(512),
    validation_data=reviews_validate.batch(512),
    epochs=15
)


model.evaluate(reviews_test.batch(512))

prediction = model.predict(reviews_test.batch(512))


for i, val in enumerate(reviews_test.take(10)):
  print(val[0])
  if prediction[i][0]>0.5:
      print("Positive Sentiment with probability: --> ", prediction[i][0])
  else:
      print("Negative Sentiment with probability: --> ", 1 - prediction[i][0])