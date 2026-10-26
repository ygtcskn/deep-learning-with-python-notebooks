"""Small, offline checks for chapter 2/3 course errata. No notebook edits."""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras
from keras import ops

checks = []

x = tf.zeros(shape=())
with tf.GradientTape() as tape:
    y = 2 * x + 3
checks.append(("ch2 cell107 unwatched tensor yields None", tape.gradient(y, x) is None))

with tf.GradientTape() as tape:
    tape.watch(x)
    y = 2 * x + 3
checks.append(("ch2 cell107 tape.watch repair yields 2", float(tape.gradient(y, x)) == 2.0))

class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(shape=(input_size, output_size), initializer="uniform")
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):
        x = ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def weights(self):
        return [self.W, self.b]

model = NaiveDense(2, 2, activation=ops.softmax)
sample_inputs = tf.ones((3, 2))
sample_labels = tf.constant([0, 1, 0])
with tf.GradientTape() as tape:
    predictions = model(sample_inputs)
    average_loss = ops.mean(ops.sparse_categorical_crossentropy(sample_labels, predictions))
gradients = tape.gradient(average_loss, model.weights)
checks.append(("ch2 Keras Variable sources work directly with tape.gradient", all(g is not None for g in gradients)))
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
optimizer.apply_gradients(zip(gradients, model.weights))
checks.append(("ch2 apply_gradients consumes zip", True))

class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        batch_dim, input_dim = input_shape
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

try:
    SimpleDense(3, activation="relu")(tf.ones((2, 4)))
except TypeError:
    checks.append(("ch3 cell131 string activation raises TypeError", True))
else:
    checks.append(("ch3 cell131 string activation raises TypeError", False))

fixed_layer = SimpleDense(3, activation=keras.activations.get("relu"))
checks.append(("ch3 resolve activation string repair", tuple(fixed_layer(tf.ones((2, 4))).shape) == (2, 3)))
checks.append(("ch3 callable activation works", tuple(SimpleDense(3, activation=keras.ops.relu)(tf.ones((2, 4))).shape) == (2, 3)))

for label, ok in checks:
    print(("PASS " if ok else "FAIL ") + label)
assert all(ok for _, ok in checks)
print(f"Versions: Python tensor checks; TensorFlow {tf.__version__}, Keras {keras.__version__}")
