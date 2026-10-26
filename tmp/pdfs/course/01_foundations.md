# Part I | Learn to read the code

## How to use this course

You have already read chapters 2 and 3 of *Deep Learning with Python*, third edition. This course revisits their code with a different question: what is Python doing at each step? The goal is to make you comfortable reading, changing, debugging, and writing that code yourself.

The source is the two third-edition companion notebooks in the main folder of your repository. The `first_edition` and `second_edition` folders were not used. This is an original teaching companion, not a reproduction of the book's prose. It covers the Python constructs that occur in these two notebooks, plus a few small extensions needed for practice. It does not claim to cover the entire Python language.

Each source locator gives a notebook chapter, a heading, and a zero-based cell index, counting both Markdown and code cells. For example, chapter 2 cell 6 is the seventh cell in the file. These are file positions, not the changing execution numbers shown beside notebook cells. Repeated code is explained once and connected to its later uses.

**Read code actively.** Before running a block, predict its result and the type and shape of each important value. Then run it. Change one thing, predict again, and explain the result in your own words. Typing a small example from memory is more useful than copying an entire training loop.

**Example labels.** Exact excerpts retain the source's logic. Adaptations use smaller inputs, clearer names, or explicit corrections. A block described as a continuation needs the objects defined immediately before it. Framework examples require the named framework; tiny Python and NumPy examples are the easiest place to begin.

### A practical study route

| Session | Work | You should be able to do this afterward |
| --- | --- | --- |
| 1 | Part I: names, calls, containers, first model | Explain every bracket in a model definition |
| 2 | Part II: arrays and slicing | Predict shapes without running the cell |
| 3 | Part II: loops and tensor operations | Write and test a small array function |
| 4 | Part II: classes and batching | Trace an object's state across calls |
| 5 | Part II: manual training | Follow arguments, returns, and weight updates |
| 6 | Part III: TensorFlow and PyTorch | Distinguish assignment from mutation |
| 7 | Part III: JAX and custom Keras layers | Pass functions and explicit state correctly |
| 8 | Exercises, repairs, and the final lab | Rebuild a small program without copying |

Allow roughly 45-90 minutes per session, depending on how much you type and experiment. Do the exercises before looking at their solutions. If a section feels easy, prove it by predicting the output of its examples.

## The execution environment is part of the lesson

Source: both notebooks, setup cells 1-3; repository README, choosing a backend.

A notebook is a collection of cells sharing one running Python process, called a kernel. Variables survive between cell executions. If you execute a cell twice, it acts on the state left by the first execution. If you change an earlier function definition, later calls use the most recently executed definition.

The display order on screen is not proof of execution order. `NameError` often means the defining cell has not run. A mysterious wrong shape can mean a previous cell already reshaped the array. In chapter 2, `train_images` is initially three-dimensional, becomes two-dimensional after preprocessing, and later returns to three dimensions when MNIST is loaded again.

Use a fresh notebook for your exercises. For the original framework examples, use the environments described by the repository README. A useful check inside any notebook is:

```python
import sys
print(sys.executable)
print(sys.version)
```

This tells you which Python process is actually running. Installing a package in one environment does not make it available to a notebook using a different environment.

### Python syntax versus notebook syntax

```text
!pip install keras keras-hub --upgrade -q
%%backend tensorflow
# @title
```

`!` asks IPython to run a shell command. It is not valid syntax in an ordinary `.py` file. The first line asks pip to install packages; `--upgrade` and `-q` are command-line options, not Python operators. `%%backend` is a custom cell magic registered by the notebook's helper. It is also not ordinary Python. The `# @title` line is a Python comment with additional meaning to Colab's interface. These setup conveniences are separate from the language you are learning.

The notebook package name `keras-hub` and Python import names are also different kinds of name. A hyphen is allowed in a package distribution name; in Python code, `keras-hub` would mean subtraction. You do not need KerasHub features for the core examples taught here.

### Selecting the Keras backend

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
```

`os` is a standard-library module. `os.environ` is a mapping of environment variable names to string values. The square brackets select a key; assignment stores the string `"tensorflow"` for that key. This configuration must happen before importing Keras in a fresh process. Changing the string after Keras has initialized does not switch the already-running backend. Restart the kernel when switching.

The local chapter 2 notebook selects TensorFlow; chapter 3 initially selects JAX. Importing `tensorflow` directly and selecting TensorFlow as Keras's backend are related but separate actions. The examples in chapter 3 also use each framework directly.

## Names, values, types, and punctuation

Source: chapter 2, first example, cells 6-21; small examples below are teaching adaptations.

```python
epochs = 5
learning_rate = 1e-3
activation_name = "relu"
use_bias = True
activation = None
```

`=` binds a name to a value. The values above have types `int`, `float`, `str`, `bool`, and `NoneType`. `1e-3` means 0.001; it is one floating-point literal. `None` is the single Python object used to represent absence, such as an optional activation that was not provided. It is not the string `"None"`, zero, or an empty list.

Names are case-sensitive: `W` and `w` can refer to different objects. Underscores connect words in names such as `train_images`. Python does not require you to declare a variable's type, and a name can later be rebound to a different type. That freedom is useful, but reusing `model` for several unrelated functions and objects can make a long notebook harder to understand.

```python
pixels = 28 * 28
print(pixels, type(pixels))
print(5 / 2, 5 // 2, 5 % 2, 2 ** 3)
```

The first line computes 784 before assigning it. The second print displays values equivalent to `2.5 2 1 8`: true division, floor division, remainder, and exponentiation. `//` is introduced here for practice; the source batch class uses `math.ceil` instead. Multiplication and division bind more tightly than addition and subtraction. Parentheses make grouping explicit: `128 * (n + 1)` differs from `128 * n + 1`.

### A punctuation decoder

| Form | Meaning in this course |
| --- | --- |
| `name = value` | Bind a name; on an attribute or index, request an update |
| `a == b` | Compare values; arrays can produce an array of answers |
| `obj.attr` | Look up an attribute on an object |
| `f(x)` | Call a function or another callable object |
| `[a, b]` | Build a list |
| `(a, b)` | Build a tuple |
| `x[i]` | Select an item or array position |
| `x[start:stop]` | Slice, with stop excluded |
| `{"loss": values}` | Build a dictionary with a key and value |
| `def f(...):` | Start a function body, defined by indentation |
| `f"loss={loss:.2f}"` | Insert and format a value inside a string |

The same character can have different roles. A colon in a slice separates bounds; a colon after `if` starts an indented block; a colon inside an f-string introduces a format. Learn the surrounding pattern rather than assigning one meaning to punctuation everywhere.

### Lists, tuples, and dictionaries

```python
units = [512, 10]
shape = (28, 28)
one_axis = (10,)
history = {"loss": [0.8, 0.4, 0.2]}
print(units[0], shape[-1], history["loss"][-1])
```

The output is `512 28 0.2`. Lists and tuples preserve order. Indexing starts at zero; `-1` selects the last item. Lists can have their contents changed, while a tuple's item references cannot be reassigned. The comma makes a one-element tuple: `(10,)` is a tuple, but `(10)` is just the integer 10 surrounded by grouping parentheses. An empty tuple is `()`; it represents a scalar array's shape later in the course.

A dictionary maps keys to values. Here the key is the string `"loss"` and its value is a list. Reading `history["loss"][-1]` means first retrieve that list, then retrieve its final item. A missing dictionary key raises `KeyError`. In `history.history`, the first `history` in the chapter 3 notebook is an object name, while the second is its attribute, which contains a dictionary.

### Names point to objects

```python
first = [1, 2]
second = first
second[0] = 99
print(first)              # [99, 2]
second = [7, 8]
print(first)              # [99, 2]
```

The assignment `second = first` does not copy a list. Both names initially refer to the same object, so changing an item through either name affects that object. Later, `second = [7, 8]` binds the name to a different object. It does not replace the object still referred to by `first`. This distinction explains much of the chapter 3 difference between rebinding a Python name and updating a framework variable.

An array's Python type and element `dtype` answer different questions. `type(x)` might be `numpy.ndarray`; `x.dtype` might be `float32`. The first describes the container object. The second describes how numerical elements are represented.

## Imports, attributes, calls, and unpacking

```python
import numpy as np
from keras import layers
from keras.datasets import mnist
```

`import numpy as np` imports a module and binds it to the local alias `np`. `as` does not change the library itself. `from keras import layers` brings a selected name into the current namespace; it does not create the separate name `keras`. If you later write `keras.Sequential`, you still need `import keras`.

Dots navigate attributes: `keras.datasets.mnist` names something inside something else. In `x.shape`, the attribute already holds a value, so you do not call it. In `x.reshape(...)`, the attribute is a method that you call. In `np.reshape(x, ...)`, a function in a module receives `x` explicitly. The API determines which form exists.

### Trace a call from inside to outside

```python
import numpy as np
x = np.zeros((2, 3), dtype="float32")
print(x.shape)
```

First Python looks up `zeros` on the module `np`. It then evaluates the arguments from left to right, building the tuple `(2, 3)` and the string `"float32"`, and calls the function. `(2, 3)` is a positional argument: its location identifies the parameter. `dtype="float32"` is a keyword argument: its name identifies the parameter. The resulting array is bound to `x`. Finally, `print` receives `x.shape`, displaying `(2, 3)`.

The `=` in `dtype="float32"` supplies a named argument. It does not create a variable named `dtype` in your surrounding scope. Ordinary positional arguments come before keyword arguments. A parameter is a name in a function definition; an argument is the actual value supplied in a call.

### Unpack the data structure, not just the words

Exact source pattern, chapter 2 cell 6:

```python
(train_images, train_labels), (test_images, test_labels) = (
    mnist.load_data()
)
```

`mnist.load_data()` returns a pair, and each element of that pair is another pair. Python matches the structure on the left to the returned structure on the right. It assigns the four contained arrays to four names. Parentheses around the right side merely let the expression continue across lines.

This tiny stand-alone example has the same structure:

```python
(train_x, train_y), (test_x, test_y) = (("A", "B"), ("C", "D"))
print(train_x, train_y, test_x, test_y)
```

It prints `A B C D`. A mismatched number of values raises `ValueError`; unpacking does not guess which value belongs to which name. `test_loss, test_acc = model.evaluate(...)` is the same idea with one level of nesting. The number of returned results depends on the configured metrics.

## Read the first neural network as a Python program

Source: chapter 2, "A first look at a neural network", cells 13-21. The blocks below retain the source's model and data flow, with extra explanatory comments. They require Keras, a selected backend, and the MNIST data loaded in the preceding section. A first dataset load may download MNIST.

```python
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
```

Read from the inside outward. Calling `layers.Dense` constructs one layer object. Calling it again constructs a second object. Square brackets put the two objects into a list, in order. `keras.Sequential` receives that list and constructs a model object. The name `model` then refers to that object.

The strings `"relu"` and `"softmax"` are values the built-in layer knows how to interpret. Python itself has no special activation syntax. A custom class must explicitly implement any string-to-function conversion it needs; this matters in the chapter 3 `SimpleDense` example.

The final comma after the second layer is legal and makes editing easier. Open brackets allow an expression to continue across physical lines. The indentation here makes nesting readable; indentation after `def`, `if`, `for`, `class`, and `with` additionally determines program structure.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

This calls a method on the existing model. All three arguments are keyword arguments. `metrics` receives a list containing one string, not a single string. The method configures how training and evaluation work. In this context Keras's `compile` method configures the model; it is distinct from the Python built-in `compile` and the framework decorators discussed later.

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

The right side is evaluated before the left name is rebound. `28 * 28` becomes 784; the shape tuple becomes `(60000, 784)`. `reshape` changes the arrangement into rows of pixels. `astype` converts element representation. Dividing by 255 performs element-wise normalization; it is not a Python loop. The labels remain class integers, so they are not divided by 255.

Later examples replace fixed sample counts with `len(images)` so they also work on small batches. Reshape preserves the number of elements. It does not resize images or invent pixels. Part II explains the shapes and memory behavior carefully.

[[PIPELINE]]

```python
history = model.fit(
    train_images, train_labels, epochs=5, batch_size=128
)
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predicted_class = predictions[0].argmax()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc:.3f}")
```

This adaptation saves the `History` returned by `fit` and names the predicted class. The first two arguments to `fit` are positional; the remaining arguments are named. A slice `0:10` selects ten examples, indices 0 through 9. Predictions have shape `(10, 10)` here: ten examples, each with ten class scores. `predictions[0]` selects one row, and `.argmax()` returns the index of its largest score. The source's `predictions[0][7]` instead selects the score for class 7; it does not find the predicted class.

The final line is an f-string. Braces contain a Python expression. `:.3f` asks for a fixed-point number with three digits after the decimal point. It changes the displayed string, not the stored accuracy. The actual scores and accuracy depend on training; do not memorize an exact result.

### Your first prediction checks

1. In `layers.Dense(512, activation="relu")`, which argument is positional and which is named?
2. What is the difference between `(784)` and `(784,)`?
3. Does `model.compile(...)` create the variable `optimizer` in your notebook?
4. If `predictions.shape` is `(10, 10)`, what is the shape of `predictions[0]`?
5. Why do `test_images[0:10]` and `test_images[0]` have different ranks?

**Answers.** 1: `512` is positional; `activation` is named. 2: the first is an integer; the second is a one-element tuple. 3: no, it passes a keyword argument to a method. 4: `(10,)`. 5: a slice keeps the selected axis, while an integer index selects one item and removes that axis.

