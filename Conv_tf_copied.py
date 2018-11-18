import tensorflow as tf
from cnn_utils_DLCoursera import *
#mnist = tf.keras.datasets.mnist

#(x_train, y_train),(x_test, y_test) = mnist.load_data()

(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes )= load_dataset()
x_train, x_test = X_train_orig / 255.0, X_test_orig / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


conv = tf.nn.conv2d(x_train[1,:,:,:],)