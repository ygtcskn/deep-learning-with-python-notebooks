# Part II | Python inside chapter 2

The chapter turns numerical ideas into executable Python. Our goal here is to read every important construct in that code, predict its behavior, and then change it confidently. The examples below use the repository's third-edition `chapter02_mathematical-building-blocks.ipynb`. A cell number is its zero-based position in the notebook file, including Markdown cells; it is not the notebook's changing execution count. “Exact excerpt” means copied code; “teaching adaptation” means modified or newly written code that exposes the same idea.

For the numerical examples in this part, run `import numpy as np` first. TensorFlow/Keras examples additionally need the TensorFlow backend selected before importing Keras. They are identified separately. Use small arrays while learning: they let you check your interpretation without training a model.

## 6. Read an array before operating on it

Notebook locator: “Data representations for neural networks,” cells 24–38.

A NumPy array is one object containing numerical data plus metadata about its organization. `shape`, `ndim`, and `dtype` are attributes, so you read them without parentheses. By contrast, `reshape(...)` is a method: parentheses call it.

Teaching adaptation; dependency: NumPy.

```python
scalar = np.array(12)
vector = np.array([12, 3, 6, 14, 7])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(scalar.shape, scalar.ndim)
print(vector.shape, vector.ndim)
print(matrix.shape, matrix.ndim)
```

Expected output:

```text
() 0
(5,) 1
(2, 3) 2
```

`np.array(...)` constructs an array. The nested brackets in the matrix describe two rows of three elements each. The scalar has no axes, so its shape is the empty tuple `()`. The vector's `(5,)` has a comma because this is a one-element tuple. `(5)` would simply be the integer `5` in parentheses. Tensor rank means number of axes here; it is not the linear-algebra notion of matrix rank.

An axis is a direction you can index. In `(60000, 28, 28)`, axis 0 selects an image, axis 1 a row, and axis 2 a column. The total number of values is the product of the shape entries. A rank-3 array need not represent three-dimensional physical space: the first axis could mean examples. Python and NumPy do not know the meaning of these axes; your program supplies it.

`len(array)` asks for the length along its first axis. For that image array it returns `60000`, not the number of pixels. `array.size` counts all elements, and `array.ndim` counts axes. `len(scalar)` fails because a rank-0 array has no first axis. This distinction prevents many incorrect loops.

The original MNIST images have dtype `uint8`: unsigned eight-bit integers ranging from 0 to 255. A dtype is the storage and arithmetic type of every element in the array; it is different from the Python type of the array object itself. `type(train_images)` is an array class, while `train_images.dtype` describes its elements. A list can freely mix objects; a regular numerical array uses a common dtype.

Notebook locator: “A first look at a neural network,” cell 15, repeated in cell 90.

Exact excerpt:

```python
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
```

First, Python calculates `28 * 28`. The reshape then groups the existing values into 60,000 rows of 784 features. Next, `astype` creates a floating-point conversion; division scales pixel values. The assignment rebinds `train_images` to the result. It does not somehow change every earlier variable that referred to the original array. The code uses a fixed dataset size; a reusable version is `images.reshape((len(images), -1))`, where `-1` asks NumPy to infer one dimension.

## 7. Indexing, slices, and shared storage

Before slicing batches, cell 39 displays one original MNIST image:

```python
import matplotlib.pyplot as plt

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

This requires Matplotlib and the original image array with shape `(60000, 28, 28)`, before flattening. `digit` has shape `(28, 28)`. `plt.cm.binary` retrieves a colormap object through two attribute lookups; `cmap=` passes that object as a keyword argument. It is not a string or a function call. `plt.imshow` draws the array as an image, and `plt.show()` displays the figure. Cell 40 reads `train_labels[4]`, using the same index to inspect its label. The index pairs an image with its answer; it is not itself the digit's class.

Notebook locator: “Manipulating tensors in NumPy,” cells 42–46; “The notion of data batches,” cells 48–50.

Python positions begin at zero. `images[4]` selects the fifth image and removes the example axis. `images[4:5]` selects a slice containing one image and preserves that axis. A slice `start:stop` includes `start` but excludes `stop`. This makes its length easy to calculate: `100 - 10 = 90` when both positions are within the array.

The chapter's three expressions `train_images[10:100]`, `train_images[10:100, :, :]`, and `train_images[10:100, 0:28, 0:28]` select the same region of the original MNIST array. A bare `:` means all entries on that axis. Commas separate axes inside a single indexing operation.

Teaching adaptation; dependency: NumPy.

```python
images = np.arange(3 * 4 * 4).reshape((3, 4, 4))
print(images[0].shape)
print(images[0:1].shape)
print(images[:, 2:, 2:].shape)
print(images[:, 1:-1, 1:-1].shape)
```

Expected shapes are `(4, 4)`, `(1, 4, 4)`, `(3, 2, 2)`, and `(3, 2, 2)`. `np.arange` creates consecutive values; here the values help you track where each slice came from. Negative indices count from the end. A stop of `-1` excludes the last position; `7:-7` in a 28-pixel dimension retains positions 7 through 20, producing length 14. Omitting the stop, as in `14:`, includes the final element.

You can also write `start:stop:step`. `vector[::2]` takes every second value, and `vector[::-1]` reverses the order through a view. Unlike indexing a single missing position, an oversized slice stop is normally clipped to the array boundary. That is why the last batch can be shorter without special slicing code.

The crucial hidden detail is that basic NumPy slicing usually returns a view: a different array object sharing the original storage. Changing a view can change its source. Assignment of a name never promises an independent copy.

Teaching adaptation; expected final array: `[0, 99, 2, 3, 4]`.

```python
original = np.arange(5)
view = original[1:4]
view[0] = 99
independent = original[1:4].copy()
independent[0] = -1
print(original)
```

`view[0]` maps to `original[1]`. Calling `.copy()` creates separate storage. The chapter uses this deliberately in its naïve functions: they modify a working copy and return it, leaving the caller's array intact. Advanced indexing with an integer list or a Boolean mask generally produces a copy instead; do not extend the basic-slice rule to all indexing. Use `np.shares_memory(a, b)` to investigate an actual pair.

For batches, the expression `images[size * n : size * (n + 1)]` selects batch number `n`, with `n=0` selecting the first batch. In cell 50, `n=3` and size 128 select positions 384 through 511. The arithmetic belongs inside the brackets because it calculates slice boundaries before indexing occurs.

## 8. Functions, loops, and vectorized operations

Notebook locator: “Element-wise operations,” cells 58–61.

Exact excerpt, cell 58:

```python
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
```

`def` binds a function object to the name `naive_relu`; its body runs only when called. The parameter `x` is a local name bound to the supplied object. `assert` checks a condition and raises `AssertionError` if it is false. Here, it documents the teaching assumption that the input is a matrix. Assertions can be disabled with optimized Python execution, so public input validation should raise an explicit exception instead.

After `x = x.copy()`, the local name points at a fresh array. `range(n)` produces integers from zero through `n - 1` as the loop requests them. The outer loop visits rows; the inner loop visits every column for each row. Indentation defines which statements belong to each loop. `return x` is outside both loops, so the function returns only after all elements have been processed. Put it inside a loop accidentally and you stop early.

Built-in `max(a, b)` chooses the greater of two scalar values here. NumPy's `np.maximum(array, 0)` instead compares element by element across an entire array. Do not substitute `max(array, 0)`: built-in scalar-oriented comparisons and array comparisons have different semantics.

In `naive_add`, cell 59, the second assertion is `x.shape == y.shape`; tuple equality gives one Boolean result. The indexed statement `x[i, j] += y[i, j]` adds into an element of the copy. For array objects, `x += y` generally modifies existing storage, while `x = x + y` computes a result and rebinds the local name. Dtypes and broadcasting can also make an in-place operation stricter.

Teaching adaptation; dependency: NumPy and `naive_relu` above.

```python
values = np.array([[-2, 3], [0, -4]])
result = naive_relu(values)
print(result.tolist())
print(values.tolist())
assert np.array_equal(result, np.maximum(values, 0))
```

Expected lists are `[[0, 3], [0, 0]]` and `[[-2, 3], [0, -4]]`. `.tolist()` makes the printed structure familiar; it is not needed for computation. `np.array_equal` yields one Boolean. By contrast, `result == values` yields an array of elementwise Booleans. Python cannot generally decide whether such an array should count as one true-or-false condition.

The chapter times 1,000 repetitions with `time.time()`, saving a start time and subtracting it afterward. `_` in `for _ in range(1000)` is a conventional name for an unused loop variable; Python still assigns it normally. The chapter's `"Took: {0:.2f} s".format(elapsed)` inserts argument zero as a floating-point value with two decimal places. `f"Took: {elapsed:.2f} s"` expresses the same formatting directly.

For elapsed-time measurements prefer `time.perf_counter()`. Timing is an experiment, not a fixed expected output: machine load, initialization, and input sizes affect results. NumPy's vectorized operations typically avoid running a Python loop for every element. “Vectorized” describes how you express and execute the operation; it does not mean your input must be a rank-1 vector.

## 9. Broadcasting and matrix multiplication

Notebook locator: “Broadcasting,” cells 63–67; “Tensor product,” cells 69–73.

Broadcasting lets compatible shapes participate in elementwise operations without writing a loop or manually repeating values. Compare shape dimensions from right to left. Each pair must either match or include a size 1; missing leading dimensions act like size 1. Thus `(32, 10)` and `(10,)` combine to `(32, 10)`. A `(32,)` vector does not combine with `(32, 10)` merely because its length matches the first dimension: alignment starts at the right.

Teaching adaptation; expected rows: `[11, 22, 33]` and `[14, 25, 36]`.

```python
x = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(x + bias)
row = np.expand_dims(bias, axis=0)
repeated = np.tile(row, (2, 1))
assert row.shape == (1, 3)
assert np.array_equal(x + bias, x + repeated)
```

`expand_dims` inserts a size-one axis: `(3,)` becomes `(1, 3)` when `axis=0`. With `axis=1` it becomes `(3, 1)`, useful for different broadcasting problems. In `tile(row, (2, 1))`, repeat the row twice along axis 0 and once along axis 1. Tiling physically constructs repeated data; ordinary broadcasting does not require this duplicated array. The chapter illustrates the idea with tiling, but `x + bias` is the direct expression you usually want.

Cell 67 combines `(64, 3, 32, 10)` and `(32, 10)` using `np.maximum`. Think of the latter shape as `(1, 1, 32, 10)`: it can be compared against every corresponding block of the larger tensor. The custom function in cell 66 does the matrix/vector version by reading `y[j]` for every row `i`; there is no `i` in that lookup because every row shares the vector.

Elementwise multiplication and matrix multiplication are separate operations. `*` multiplies corresponding elements, with broadcasting. `@` invokes matrix multiplication rules. `np.matmul(x, y)` is the NumPy function form used alongside `x @ y` in cell 69. For two one-dimensional arrays it returns their dot product, a sum of products.

Teaching adaptation; dependency: NumPy.

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a * b)
print(a @ b)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix @ b)
```

Expected results are `[10, 40, 90]`, `140`, and `[140, 320]`. Matrix/vector multiplication turns shape `(rows, columns)` and `(columns,)` into `(rows,)`. Matrix/matrix multiplication turns `(m, k)` and `(k, n)` into `(m, n)`. The shared dimension is summed over. Elementwise multiplication has no such summation.

Cell 70 initializes `z = 0.0` before its loop and accumulates `z += x[i] * y[i]`. An accumulator must be initialized before use and outside the loop it summarizes. If `z = 0.0` were inside the loop, only the final product would remain. Cell 71 uses `np.zeros(x.shape[0])` to create one accumulator per output row. Its default dtype is floating point.

Cell 72 defines `naive_matrix_vector_product` a second time. This replaces the earlier binding of that function name; it does not overload the function based on argument types. The replacement calls `naive_vector_product(x[i, :], y)` for each row. Cell 73 extends this pattern: `x[i, :]` supplies a row, `y[:, j]` supplies a column, and their dot product becomes `z[i, j]`.

## 10. Reshape and transpose answer different questions

Notebook locator: “Tensor reshaping,” cells 75–79.

Reshaping changes how an ordered collection of values is grouped into axes. The total element count must stay the same. Transposing changes axis order. These can produce the same output shape but different element arrangements.

Teaching adaptation; dependency: NumPy.

```python
x = np.array([[0., 1.], [2., 3.], [4., 5.]])
print(x.reshape((2, 3)))
print(x.T)
```

The reshape produces rows `[0, 1, 2]` and `[3, 4, 5]`. The transpose produces `[0, 2, 4]` and `[1, 3, 5]`. Both shapes are `(2, 3)`, so checking shape alone would not catch a confusion between these operations. The decimal points in `0.` and `1.` denote floating-point literals.

`x.T` is the convenient matrix-transpose attribute; `np.transpose(x)` is the chapter's function form. For higher-rank arrays, transpose can take an explicit axis permutation. The default reverses the axis order. A rank-1 array stays rank 1 after `.T`; turning a vector into a column requires an explicit size-one axis.

Reshape may return a view or need a copy depending on the layout. Do not rely on reshape to isolate your data, and do not assume every reshape is free. If independent storage matters, request a copy explicitly. Neither operation normalizes values or changes the meaning of labels; those are separate tasks.

## 11. Classes make state and behavior travel together

Notebook locator: “A simple Dense class,” cell 96; “A simple Sequential class,” cells 98–99.

`class NaiveDense:` defines a new Python class. Calling `NaiveDense(...)` creates an instance and initializes it through `__init__`. Python supplies the instance as the first method argument, conventionally named `self`. `self.W` belongs to that instance; an ordinary local variable such as `x` exists only in the method call. Two layer instances therefore retain different weights.

The following is a teaching adaptation of cell 96, compressed without changing its structure. Dependencies: Keras with a configured backend, `from keras import ops`.

```python
class NaiveDense:
    def __init__(self, input_size, output_size, activation=None):
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer="uniform"
        )
        self.b = keras.Variable(shape=(output_size,), initializer="zeros")

    def __call__(self, inputs):
        x = ops.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    @property
    def weights(self):
        return [self.W, self.b]
```

The initializer strings tell Keras how to create numerical weights. Python handles argument passing and attribute assignment; Keras supplies numerical storage and trainable variables. `(output_size,)` is again a one-element shape tuple. `None` is a special singleton used here to mean “no activation.” `is not None` checks identity against that singleton and is the normal Python idiom.

`activation=ops.relu` passes the function object so the layer can call it later. `activation=ops.relu(x)` would call it immediately and pass the result instead. The expression `self.activation(x)` is the later invocation. Functions are values: you can store them in variables, pass them as arguments, and retain them in attributes.

Defining `__call__` makes an instance callable. After `layer = NaiveDense(3, 2)`, `layer(inputs)` invokes its `__call__` method. Construction and prediction are different calls: `NaiveDense(...)` initializes state; `layer(...)` uses that state. For inputs shaped `(5, 3)`, the matrix multiplication produces `(5, 2)`, and the bias of shape `(2,)` broadcasts across five examples.

`@property` is decorator syntax. It transforms the `weights` method into an attribute-style interface. You write `layer.weights`, without parentheses; reading it executes the getter and returns a newly created Python list containing references to the two variables. The variables themselves are not copied. `layer.weights()` would try to call the returned list and fail.

Cell 98 stores a list of layers in `self.layers`. Its `for layer in self.layers` loop repeatedly replaces local `x` with `layer(x)`, passing each result into the next layer. It is function composition expressed with ordinary iteration. Its weights property starts with `weights = []` and executes `weights += layer.weights` for every layer.

Here is the complete source class, so you can assemble the manual model:

```python
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
```

For Python lists, `+=` extends the current list with the items from another iterable. It does not add corresponding numerical values. `weights.append(layer.weights)` would instead append an entire list as one item, producing nested lists. Two dense layers each expose two variables, so the flattened list has length four, matching `assert len(model.weights) == 4` in cell 99. Keras classes in real applications add automatic tracking and many conveniences; this plain class teaches the Python mechanics.

## 12. Batches are a lesson in mutable state

Notebook locator: “A batch generator,” cell 101.

The chapter's `BatchGenerator` stores arrays, a batch size, an index initially equal to zero, and `num_batches = math.ceil(len(images) / batch_size)`. `math.ceil` rounds upward: 10 examples with batches of 4 require three batches. `/` performs division; the result here is passed to `ceil`. The integer-only alternative is `(count + size - 1) // size`, using floor division.

Each call to its ordinary `.next()` method slices both arrays from the current index, increases `self.index`, and returns `images, labels`. Returning comma-separated values packages them as a tuple; `images_batch, labels_batch = ...` unpacks that tuple. Changing `self.index` affects the next call because instance attributes persist.

Important distinction: calling a method named `.next()` does not make an object a Python iterator. Python's `next(obj)` and `for ... in obj` use special methods `__next__` and `__iter__`. The chapter's class does not define those. It also does not raise `StopIteration` after its final batch; additional `.next()` calls yield empty slices. Its training loop avoids that by calling it exactly `num_batches` times.

The source class is short enough to trace in full. Assume matching image and label counts and a positive batch size:

```python
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index:self.index + self.batch_size]
        labels = self.labels[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels
```

The default `batch_size=128` is used when the caller omits that argument. Defaults are evaluated when the function definition executes, not on each call. An integer default is safe here; mutable defaults such as a list can accidentally share state across calls. `images` inside `next` is a local slice; `self.images` remains the full stored array. The similarly named objects have different scopes and roles.

This separate teaching adaptation demonstrates the iterator protocol; dependency: NumPy for the example arrays. It is not a replacement required by the chapter.

```python
class Batches:
    def __init__(self, images, labels, size=4):
        if len(images) != len(labels) or size <= 0:
            raise ValueError("Check lengths and positive batch size")
        self.images, self.labels = images, labels
        self.size, self.index = size, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.images):
            raise StopIteration
        start = self.index
        self.index += self.size
        return (self.images[start:self.index],
                self.labels[start:self.index])
```

With `Batches(np.arange(10), np.arange(10))`, a loop receives sizes 4, 4, and 2. Once consumed, this particular iterator remains exhausted. Create a new instance for a new epoch. A reusable iterable and an iterator are related but different interfaces; returning `self` implements the simple single-pass iterator pattern.

## 13. Gradients and the training loop: Python wiring

Notebook locator: “The weight update step,” cells 104–105; “Gradient computation,” cells 107–108; “The full training loop,” cells 110–111.

In cell 104, `learning_rate = 1e-3` is scientific notation for `0.001`. `zip(gradients, weights)` produces pairs, and `for g, w in ...` unpacks each pair. Pair order matters: every gradient must be paired with the variable it describes. Ordinary `zip` stops at the shorter input; it does not flag unequal lengths. For known lists, compare lengths first, or use `zip(..., strict=True)` on Python 3.10 or newer.

`w.assign(w - g * learning_rate)` updates the contents of a backend variable. Writing `w = w - g * learning_rate` would merely rebind the loop's local name to a result. It would not perform the intended variable update. Cell 105 defines `update_weights` again using an optimizer; subsequent calls resolve to this newer definition. That function accesses `optimizer` from its surrounding module/notebook namespace.

The notebook contains a small trap in cell 107: `x = tf.zeros(shape=())` creates a constant scalar tensor. A gradient tape automatically watches trainable TensorFlow variables, but it does not automatically watch this constant. Consequently, the original demonstration yields `None` for its gradient. Here is an explicitly corrected teaching version; dependency: TensorFlow.

```python
import tensorflow as tf

x = tf.zeros(shape=())
with tf.GradientTape() as tape:
    tape.watch(x)
    y = 2 * x + 3
gradient = tape.gradient(y, x)
print(float(gradient))
```

Expected result: `2.0`. `with` enters and exits a context manager, which here establishes recording for the indented operations. `as tape` binds the resulting tape object. Calling `.gradient` happens after the block in this example. `None` means a gradient was not available; it is not a numerical zero. Replacing the constant with a suitable `tf.Variable` is another way to make this example differentiable.

Cell 108 calls the model, obtains a vector of per-example losses, averages it to one scalar, differentiates that scalar with respect to `model.weights`, updates the weights, and returns the average loss. The order matters. Python constructs the control flow; the tensor framework records differentiable numerical operations. Converting intermediate tensors to Python numbers inside that computation would interrupt the connection needed for gradients.

To connect the classes above, use these source-based definitions in a fresh TensorFlow-backend Keras session. They continue the `NaiveDense` and `NaiveSequential` definitions; the notebook-only `%%backend` wrapper is omitted.

```python
import keras
from keras import ops

learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign(w - g * learning_rate)

def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        losses = ops.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = ops.mean(losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss
```

```python
model = NaiveSequential([
    NaiveDense(28 * 28, 512, activation=ops.relu),
    NaiveDense(512, 10, activation=ops.softmax),
])
assert len(model.weights) == 4
```

The model initialization needs `keras` and `ops` available when the constructors execute. The TensorFlow import appears in the tape example above. Alternatively, cell 105 creates `optimizer = keras.optimizers.SGD(learning_rate=1e-3)` and replaces the update body with `optimizer.apply_gradients(zip(gradients, weights))`. Both versions need the gradient-variable pairs in corresponding order. Choose one update function before running the training loop.

In cell 110, the `fit` signature accepts `batch_size`, but the original body constructs `BatchGenerator(images, labels)` without forwarding it. Therefore its parameter has no effect unless it equals the generator's default. The corrected teaching version below preserves the chapter's interface. Dependencies: the original `BatchGenerator` from cell 101, a compatible model, and `one_training_step` from cell 108.

```python
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter + 1}")
        batches = BatchGenerator(images, labels, batch_size)
        for batch_counter in range(batches.num_batches):
            images_batch, labels_batch = batches.next()
            loss = one_training_step(
                model, images_batch, labels_batch
            )
            if batch_counter % 100 == 0:
                value = float(loss)
                print(f"loss at batch {batch_counter}: {value:.2f}")
```

The new batch generator inside each epoch resets traversal. `%` computes a remainder; `% 100 == 0` logs batches 0, 100, 200, and so on. The displayed epoch adds one for human numbering while the loop remains zero-based. Converting the completed loss to a Python float here is for eager TensorFlow logging after gradient computation; it is not part of the differentiable calculation. This simple teaching loop does not shuffle examples or provide the features of Keras `Model.fit`.

Notebook locator: “Evaluating the model,” cell 113. Predictions have one row per example and one column per class. `argmax(..., axis=1)` removes the class axis by returning the index of each row's greatest value. With no axis, NumPy's `argmax` instead searches the flattened array. Array equality produces Boolean matches; explicitly cast those Booleans before averaging for robust backend behavior.

Corrected teaching adaptation; dependencies: Keras ops and trained model.

```python
predictions = model(test_images)
predicted_labels = ops.argmax(predictions, axis=1)
matches = ops.equal(predicted_labels, test_labels)
accuracy = ops.mean(ops.cast(matches, "float32"))
print(f"accuracy: {float(accuracy):.2f}")
```

True becomes 1 and false becomes 0, so the mean is a fraction correct. An accuracy of `0.96` is a proportion; format it as `{float(accuracy):.1%}` to display `96.0%`. A bare f-string in a notebook displays because the notebook renders its final expression; in a `.py` file you need `print` to show it.

## 14. Practice before reading the solutions

These exercises use tiny inputs so you can verify every value. Predict first, execute second, then explain any mismatch in one sentence.

1. **Shape detective.** Create `a = np.arange(24).reshape((2, 3, 4))`. Predict `a.ndim`, `len(a)`, `a.size`, and the shapes of `a[1]`, `a[1:2]`, and `a[:, :, -1]`. Which expression preserves the example axis?
2. **Mutation audit.** Create `a = np.arange(6)`, then `b = a[1:4]` and `c = b.copy()`. Assign `b[1] = 90` and `c[0] = -5`. Predict all three arrays. Explain why creating a new variable did not isolate `b`.
3. **Broadcast repair.** For `x = np.arange(12).reshape((3, 4))`, add `[10, 20, 30]` to its rows so the first row gains 10, the second 20, and the third 30. Avoid tiling. State all operand shapes.
4. **Write a numerical function.** Implement `row_sums(x)` with a loop and an accumulator array. Reject non-matrices with `ValueError`. Return one total per row without modifying `x`. Compare against `x.sum(axis=1)` for positive and negative values.
5. **Multiplication and reshape.** With `x = np.array([[1, 2], [3, 4]])` and `v = np.array([10, 20])`, calculate `x * v` and `x @ v` by hand. Then compare `x.reshape((4, 1))` with `x.T.reshape((4, 1))` and explain the changed order.
6. **Callable object.** Write a `Shift` class that stores an amount and returns its input plus that amount when called. Give it a read-only `description` property. Create two objects with different amounts and show their state is independent.
7. **Batch boundary.** Use the `Batches` iterator above with ten examples, matching labels, and size 4. Record batch lengths and label values. Verify a second loop over the same object yields nothing. Explain why a training epoch should create a new instance.
8. **Training-code review.** Explain the fixes for the constant-tensor gradient example and the unused `batch_size` parameter. Then compute accuracy from predictions `[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]` with labels `[1, 0, 0]`, using NumPy. State the unrounded accuracy and the two-decimal display.

## Worked solutions for Part II

**1. Shape detective.** Rank is 3, first-axis length is 2, and size is 24. The three slice shapes are `(3, 4)`, `(1, 3, 4)`, and `(2, 3)`. `a[1:2]` preserves an example axis of length one; `a[1]` removes it. The final expression removes the last axis because `-1` is one integer index.

**2. Mutation audit.** `a` becomes `[0, 1, 90, 3, 4, 5]`, `b` becomes `[1, 90, 3]`, and `c` becomes `[-5, 2, 3]`. `c` copied the values before the changes. `b` is a view into `a`, so its element 1 is `a`'s element 2. Separate names and separate storage are different concepts.

**3. Broadcast repair.** Insert a size-one column axis so `(3, 1)` broadcasts over the four columns of `(3, 4)`.

```python
x = np.arange(12).reshape((3, 4))
offsets = np.array([10, 20, 30])[:, None]
result = x + offsets
assert result.tolist() == [
    [10, 11, 12, 13],
    [24, 25, 26, 27],
    [38, 39, 40, 41],
]
```

`None` inside NumPy indexing inserts a new axis; it is equivalent here to `np.expand_dims(..., axis=1)`. This is a new indexing use of the same Python singleton.

**4. A loop with one accumulator per row.** The float output is deliberate for this teaching version. Real numerical utilities should choose their dtype according to precision and overflow requirements.

```python
def row_sums(x):
    if x.ndim != 2:
        raise ValueError("Expected a matrix")
    totals = np.zeros(x.shape[0], dtype="float64")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            totals[i] += x[i, j]
    return totals

x = np.array([[1, -2, 3], [-4, 5, 6]])
before = x.copy()
assert np.array_equal(row_sums(x), [2, 7])
assert np.array_equal(row_sums(x), x.sum(axis=1))
assert np.array_equal(x, before)
```

**5. Different operations.** `x * v` produces `[[10, 40], [30, 80]]`. `x @ v` produces `[50, 110]`. The flattened value orders are `[1, 2, 3, 4]` and `[1, 3, 2, 4]`, respectively. Transpose exchanges rows and columns before reshape reads the values in row order.

**6. State plus a callable interface.** No numerical framework is necessary here.

```python
class Shift:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, value):
        return value + self.amount

    @property
    def description(self):
        return f"Add {self.amount}"

small = Shift(2)
large = Shift(10)
assert small(3) == 5
assert large(3) == 13
assert small.description == "Add 2"
```

The property has no setter, so assigning directly to `small.description` raises `AttributeError`. This does not make the whole object immutable: `small.amount` is still an ordinary writable attribute.

**7. Exhaustion is state.**

```python
iterator = Batches(np.arange(10), np.arange(10), size=4)
seen = [(len(x), y.tolist()) for x, y in iterator]
assert seen == [
    (4, [0, 1, 2, 3]),
    (4, [4, 5, 6, 7]),
    (2, [8, 9]),
]
assert list(iterator) == []
```

This list comprehension evaluates its expression for each pair from the iterator. After the three batches, the index is past the final example, so `__next__` raises `StopIteration`. A new object restores an index of zero.

**8. Review and accuracy.** Explicitly watch the constant tensor inside the tape, or use a suitable trainable variable. Forward `batch_size` when constructing the batch generator. For the accuracy calculation:

```python
probabilities = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
labels = np.array([1, 0, 0])
predicted = probabilities.argmax(axis=1)
accuracy = (predicted == labels).astype("float32").mean()
print(predicted.tolist())
print(f"{accuracy:.2f}")
```

Predicted classes are `[1, 0, 1]`; two of three match. The mathematical fraction is `2/3`; the float32 representation is approximately `0.6666667`, and the two-decimal display is `0.67`. Formatting changes the displayed text, not the stored value.
