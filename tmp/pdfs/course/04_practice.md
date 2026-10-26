# Part IV | Read, repair, and rebuild

## Decode the notebook's backend helper

Source: cell 3 in both notebooks. This helper is easy to skip, but it contains several useful Python patterns. The version below preserves its behavior and splits a long assignment for readability. It runs inside IPython/Jupyter; `get_ipython()` is supplied by that environment.

```python
import os
from IPython.core.magic import register_cell_magic

@register_cell_magic
def backend(line, cell):
    current = os.environ.get("KERAS_BACKEND", "")
    required = line.split()[-1]
    if current == required:
        get_ipython().run_cell(cell)
    else:
        print(
            f"This cell requires the {required} backend. "
            f"Set KERAS_BACKEND to \"{required}\", "
            "restart the runtime, and rerun the notebook."
        )
```

`def` creates a function object; it does not immediately execute the function body. `@register_cell_magic` applies a decorator that registers the function with IPython. When a user runs a cell beginning with `%%backend tensorflow`, IPython supplies the text after the magic name as `line`, and the remainder of the cell as the string `cell`. The decorator is an API provided by IPython; the decorator syntax itself belongs to Python.

`os.environ.get("KERAS_BACKEND", "")` looks up a key and returns an empty string if that key is missing. Unlike `os.environ["KERAS_BACKEND"]`, it avoids a `KeyError` for an absent setting. The empty string is a default value, not an exception handler.

`line.split()` splits a string on whitespace and returns a list of words. `[-1]` selects its last word. For `line = "tensorflow"`, the list is `["tensorflow"]`. If the string has no words, indexing the empty list raises `IndexError`. The helper assumes you supply a backend name.

`if current == required:` compares two strings. The indented body executes only if they compare equal. `else:` handles the other case. `get_ipython()` returns an object; `.run_cell(cell)` calls a method on that object. This evaluates the cell's text in the interactive environment. It does not change Keras's backend by itself.

The three adjacent strings inside `print(...)` form a single string. Python joins adjacent string literals, even across physical lines inside parentheses. Each `f` prefix applies only to its own literal. `\"` embeds a literal double quote inside a double-quoted string. An alternative would be a single-quoted outer string containing double quotes.

The source writes `current, required = expression1, expression2` on one line. That is the same unpacking pattern you saw with MNIST. Our adaptation uses two assignments to make evaluation easier to follow. The comment `# @title` in the source does not decorate the function; the non-comment line starting with `@register_cell_magic` does.

**Micro-exercise.** Predict `"  use tensorflow  ".split()[-1]`. Then explain why `"   ".split()[-1]` fails. Answer: the first produces `"tensorflow"`; the second tries to read the last item of an empty list.

## A debugging routine you can reuse

When a cell fails, read the final exception message first, then trace back to the line in your code that caused it. Do not respond by rerunning random cells. Check what values the failing expression actually received.

```python
print(type(inputs))
print(inputs.shape)
print(inputs.dtype)
print(type(model))
```

These checks distinguish a NumPy array from a framework tensor, a two-dimensional matrix from a one-dimensional vector, and a function from a model instance. Add them immediately before the failing line. Remove or reduce them when you understand the issue.

| Symptom | Likely Python issue | First useful check |
| --- | --- | --- |
| `NameError` | A name was never bound in this kernel | Run its definition or import cell |
| Missing positional arguments | The call does not match the definition | Compare the call with `def` |
| `str` object is not callable | Code uses a string as if it were a function | Inspect the activation's value and type |
| `AttributeError` | Object has no requested attribute | Check the module and spelling |
| Cannot reshape array | Element counts do not match | Compare old size with new shape product |
| Broadcasting error | Trailing dimensions do not align | Write both shapes side by side |
| Ambiguous array truth value | `if` received many Booleans | Choose an explicit `.any()` or `.all()` |
| Gradient is `None` | The framework did not track a path | Check variable status, watch, and graph |
| Loss changes unexpectedly on rerun | The kernel retained model or gradient state | Recreate the intended starting state |

`help(function)` displays documentation for the function object. `type(object)` identifies its type. `dir(object)` lists available names and is useful for exploration, though it can be long. Read the signature before changing a call: a positional tuple can accidentally be bound to the wrong parameter, as in the chapter 3 noise example.

### A shape mismatch that does not raise an error

```python
import numpy as np
targets = np.array([0.0, 1.0, 0.0])
predictions = np.array([[0.1], [0.9], [0.2]])
wrong = targets - predictions
right = targets.reshape((-1, 1)) - predictions
print(wrong.shape, right.shape)
```

This prints `(3, 3) (3, 1)`. Broadcasting can create a valid array with the wrong meaning. For a per-example loss, you want each prediction paired with its matching target. Shape checks belong to understanding the program, not just fixing exceptions. `-1` in a reshape asks NumPy to infer that dimension from the number of elements.

### Assertions versus validation

The source uses `assert` to express assumptions while teaching, such as requiring a matrix. An assertion that is false raises `AssertionError`. Python can disable assertions under optimized execution, so use explicit validation for a reusable public function.

```python
def require_positive_batch_size(batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return batch_size
```

This is a small extension beyond the notebook. `raise` deliberately reports an invalid value. The message explains what the caller must fix. An exception interrupts the normal path; `return` is reached only for acceptable input.

## Repair cards for the supplied notebooks

These observations apply to the local third-edition files inspected for this course. The original notebooks were left unchanged. Each repair is explained in the relevant lesson as well.

| Source | What needs attention | Repair |
| --- | --- | --- |
| Ch. 2 cell 107 | A constant is not watched by the tape | Call `tape.watch(x)` inside the context |
| Ch. 2 cell 110 | `fit` accepts `batch_size` but does not forward it | Pass it into `BatchGenerator` |
| Ch. 3 cells 61, 79 | `torch.nn.relu` is not the functional API | Use `torch.nn.functional.relu` |
| Ch. 3 cell 70 | Functional `model` needs three arguments | Call `model(inputs, W, b)` |
| Ch. 3 cell 90 | `(3,)` is passed as the distribution mean | Use `size=(3,)` or `size=x.shape` |
| Ch. 3 cells 126, 131 | Custom layer calls activation strings | Resolve with `keras.activations.get` |

Two further habits improve clarity. The TensorFlow training step calls mean squared error with predictions and targets reversed; squaring the difference makes that particular result the same, but consistent argument order prevents bugs with asymmetric losses. The chapter 2 accuracy example reduces Boolean matches; explicitly casting matches to a floating dtype makes the intended numeric mean clear across backends.

Some examples are intentionally minimal. `BatchGenerator.next()` is not Python's iterator protocol, even though the class is called a generator. A raw `model` score from the linear classifier is not automatically a probability. A timing comparison does not promise a fixed speedup on every computer. Separating Python behavior from library behavior helps you read such simplifications accurately.

## Final lab: build a tiny prediction pipeline

This lab combines the Python patterns from both chapters without downloading data or installing a deep-learning framework. It requires only NumPy. Run the following blocks in order in a fresh notebook. The deliberately chosen weights are fixed; this is a programming exercise, not a trained classifier.

### Your task before reading the solution

1. Create five tiny grayscale images and five labels. Convert the images to floating point, flatten each image, and scale pixel values into the interval from 0 to 1.
2. Build a callable class that stores a weight matrix and bias, computes `inputs @ W + b`, and exposes a read-only `weights` property returning a list.
3. Write a batch function that returns image and label slices. Batch size 2 should produce lengths 2, 2, and 1.
4. Compute predicted classes using `argmax` along the class axis. Compare with labels and return both accuracy and predictions.
5. Test that preprocessing does not change the original pixel array. Explain why a tied maximum chooses the first class here.

Expected checkpoints: features have shape `(5, 4)` and dtype `float32`; scores have shape `(5, 2)`; predicted labels are `[1, 0, 1, 0, 0]`; accuracy is 0.8. Use these checkpoints to locate a mistake, rather than replacing your entire solution.

### Worked solution, step 1: data and preprocessing

```python
import numpy as np

images = np.array([
    [[0, 255], [0, 255]],
    [[255, 0], [255, 0]],
    [[0, 128], [0, 128]],
    [[128, 0], [128, 0]],
    [[0, 0], [0, 0]],
], dtype="uint8")
labels = np.array([1, 0, 1, 0, 1])
original = images.copy()

def prepare(images):
    flat = images.reshape((len(images), -1))
    return flat.astype("float32") / 255

features = prepare(images)
assert features.shape == (5, 4)
assert features.dtype == np.float32
assert np.array_equal(images, original)
```

Each inner two-by-two list is one image. `len(images)` counts images, not pixels. Flattening preserves the leading sample axis. The dtype conversion creates a separate floating-point array, so the returned normalized values do not overwrite the original integer pixels. `np.array_equal` is a single Boolean comparison of whole-array equality, suitable for an assertion.

### Worked solution, step 2: a callable object

```python
class TinyDense:
    def __init__(self, W, b):
        self.W = W.copy()
        self.b = b.copy()

    def __call__(self, inputs):
        return inputs @ self.W + self.b

    @property
    def weights(self):
        return [self.W, self.b]

W = np.array([
    [1, -1], [-1, 1], [1, -1], [-1, 1]
], dtype="float32")
b = np.zeros((2,), dtype="float32")
model = TinyDense(W, b)
assert len(model.weights) == 2
```

`self` refers to the instance being initialized or called. Copying the supplied arrays prevents later item changes to the caller's arrays from silently changing this model. The property has no setter, but the arrays inside its returned list are still mutable. A read-only property does not make the objects it exposes immutable.

`features @ W` multiplies a `(5, 4)` matrix by a `(4, 2)` matrix. Adding the `(2,)` bias broadcasts it over rows. Calling `model(features)` invokes the instance's `__call__` method and returns a `(5, 2)` score array.

### Worked solution, step 3: slicing batches

```python
def take_batch(features, labels, start, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stop = start + batch_size
    return features[start:stop], labels[start:stop]

batch_lengths = []
for start in range(0, len(features), 2):
    x_batch, y_batch = take_batch(features, labels, start, 2)
    scores = model(x_batch)
    batch_lengths.append(len(x_batch))
    assert scores.shape == (len(y_batch), 2)

assert batch_lengths == [2, 2, 1]
```

The three-argument `range(start, stop, step)` produces 0, 2, and 4. Slicing tolerates a stop beyond the end of an array, so the last batch contains one example. A single return statement supplies a tuple of two slices, which the caller unpacks. `.append(...)` mutates the existing list and returns `None`; avoid writing `batch_lengths = batch_lengths.append(...)`.

### Worked solution, step 4: evaluate and report

```python
def evaluate(model, features, labels):
    scores = model(features)
    predicted = scores.argmax(axis=1)
    accuracy = float(np.mean(predicted == labels))
    return accuracy, predicted

accuracy, predicted = evaluate(model, features, labels)
print(predicted.tolist())
print(f"Accuracy: {accuracy:.1%}")
assert predicted.tolist() == [1, 0, 1, 0, 0]
assert np.isclose(accuracy, 0.8)
```

The output is `[1, 0, 1, 0, 0]` and `Accuracy: 80.0%`. `.tolist()` converts an array into ordinary Python containers. `float(...)` converts the scalar accuracy into a Python float. The f-string specifier `.1%` formats a fraction as a percentage with one decimal place. `np.isclose` makes a tolerance-based floating-point comparison.

The final all-zero image produces two equal scores. NumPy's `argmax` selects the first maximum, so its predicted class is 0. Its supplied label is 1; therefore exactly four of the five examples match. This lab evaluates raw class scores, so no probability interpretation is required.

### Change requests that prove understanding

Change batch size to 3 and predict the batch lengths before running it. Then set `model.b[:] = [0, 0.25]` and predict what happens to the all-zero image. Changing the external `b` alone would not work because the constructor copied it. Finally, remove the copies in `TinyDense.__init__`, mutate `W[0, 0]` after constructing the model, and explain why its behavior changes.

Answers: the lengths become `[3, 2]`; the all-zero image now predicts class 1; without copies, the model and the caller share the same weight array. Recreate the original model afterward if you want the original checkpoints to pass again.

## A compact Python reference for these chapters

| Pattern | Read it as |
| --- | --- |
| `from module import name` | Bind this exported name in the current namespace |
| `f` versus `f(x)` | Function object versus result of calling it |
| `obj.method(x)` | Call a method associated with this object |
| `self.x = x` | Save the argument on the instance |
| `return a, b` | Return one tuple containing two items |
| `a, b = result` | Unpack two items into two names |
| `if x is not None:` | Test whether an optional value is present |
| `for item in items:` | Visit each item in an iterable |
| `for g, w in zip(gs, ws):` | Pair corresponding gradients and weights |
| `for _ in range(n):` | Repeat n times, ignoring the loop value |
| `with context() as name:` | Enter a context, bind its enter-result, then exit |
| `@decorate` above `def f` | Transform or register a function at definition time |
| `class Child(Parent):` | Define a class that inherits from another |
| `super().__init__()` | Run the next initializer in inheritance order |
| `@property` | Expose a method's returned value through attribute syntax |
| `x += y` | Request augmented assignment; mutation depends on type |
| `x @ y` | Matrix multiplication, implemented by the operands' types |
| `x[i, j]` | Index two array axes in one operation |
| `x[:, 1:2]` | Keep every row and a one-column slice |
| `x[:, 1]` | Keep every row, selecting and removing one column axis |
| `x.shape`, `x.ndim`, `x.dtype` | Axis sizes, number of axes, element representation |
| `loss:.4f` inside an f-string | Display four decimal places |

Before you call a section finished, explain its code aloud without using the phrase "it just does that." Name the object, describe the operation, and state the result's type, shape, or changed state. That is the skill that transfers from these chapters to new Python programs.

