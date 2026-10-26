# Part III | Python inside chapter 3

Chapter 3 repeats a small learning problem through TensorFlow, PyTorch, JAX, and Keras. This makes it an excellent Python laboratory: the mathematics stays familiar while the way a program stores state, calls functions, and organizes objects changes. Your aim here is to read those decisions confidently.

All cell numbers below are zero-based positions in the local third-edition `chapter03_introduction-to-ml-frameworks.ipynb`, counting both Markdown and code cells. They are not execution counters. Unless marked “notebook excerpt,” examples are teaching adaptations. Run each framework section in its own fresh notebook or script, after its imports and setup.

Validation for this part: all 34 Python blocks passed syntax checks. The NumPy exercises, TensorFlow derivatives and 40-step training loop, and custom Keras layer passed local checks. Keras fitting and prediction were checked for one epoch using the TensorFlow backend. PyTorch and JAX examples were reviewed against the notebook and official APIs but were not executed because those packages were unavailable.

## 19. Recognizing what belongs to Python

Source: “First steps with TensorFlow,” cells 11–25; corresponding PyTorch cells 50–61 and JAX cells 84–101.

In `tf.ones(shape=(2, 1))`, Python supplies attribute access, function-call parentheses, a keyword argument, and a tuple. TensorFlow supplies the meaning of `ones` and `shape`. Learning to separate these layers makes an unfamiliar library much less intimidating.

```python
import tensorflow as tf

x = tf.ones(shape=(2, 1))
print(x.shape)
print(x.dtype)
```

`tf` is an ordinary name bound to an imported module. The dot in `tf.ones` retrieves an attribute from that module. Parentheses then call it. In contrast, `x.shape` reads an attribute without a call. `shape` is exposed as a property by many tensor types: its attribute-style interface may compute a result internally. You cannot infer that every attribute is merely a stored value.

The punctuation in `shape=(2, 1)` does not perform a variable assignment in the surrounding scope. Inside a call, it passes a keyword argument named `shape`. The tuple says two rows, one column. `(2, 1)` and `(2,)` describe different ranks; `(2)` is just the integer two in grouping parentheses.

Floating-point notation also matters. `0.`, `0.0`, and `0e0` are float literals. `0` is an integer literal. When differentiation is involved, use floating tensors rather than assuming the framework will differentiate integer data.

### Similar mathematics, different function signatures

| Intention | TensorFlow | PyTorch | JAX |
|---|---|---|---|
| Ones | `tf.ones(shape=(2, 1))` | `torch.ones(size=(2, 1))` | `jnp.ones(shape=(2, 1))` |
| Explicit values | `tf.constant([1., 2.])` | `torch.tensor([1., 2.])` | `jnp.array([1., 2.])` |
| Matrix product | `tf.matmul(a, b)` | `torch.matmul(a, b)` | `jnp.matmul(a, b)` |
| Join rows | `tf.concat((a, b), axis=0)` | `torch.cat((a, b), dim=0)` | `jnp.concatenate((a, b), axis=0)` |

The parameter names are API choices, not interchangeable Python synonyms. Copying `axis=0` into a call that expects `dim=0` may fail. Read the signature with `help(function)` when uncertain. The third column's concatenation example is an extension; the notebook itself does not demonstrate JAX concatenation.

In `torch.rand(3, 1)`, two positional integers specify dimensions. In the notebook's `torch.normal(mean=..., std=...)`, tensors supply elementwise means and standard deviations. Do not generalize the calling convention from one random function to another.

TensorFlow's normal sampler uses `shape`, `mean`, and `stddev`; its uniform sampler uses `shape`, `minval`, and `maxval`. A mean is the distribution's center, while standard deviation describes its spread. A uniform draw uses an interval instead. The spelling `dtype="float32"` passes a string identifier; `dtype=torch.float32` retrieves a dtype object from PyTorch. Those choices are accepted by the relevant APIs, not by a universal Python conversion rule.

The chapter's `square`, `sqrt`, and `+` examples operate element by element. `matmul` combines entries across a shared dimension. Concatenating two `(2, 2)` arrays on axis zero produces `(4, 2)`; it joins existing rows without adding a new axis. Parentheses in `(a, b)` merely package the arguments; the concatenation function decides what to do with them.

## 20. Building the shared dataset without hidden steps

Source: “An end-to-end example: A linear classifier in pure TensorFlow,” cells 34–37. Later framework sections reuse these arrays.

```python
import numpy as np

n = 1000
rng = np.random.default_rng(1337)
negative = rng.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=n
)
positive = rng.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=n
)
inputs = np.vstack((negative, positive)).astype("float32")
targets = np.vstack((
    np.zeros((n, 1), dtype="float32"),
    np.ones((n, 1), dtype="float32"),
))
assert inputs.shape == (2 * n, 2)
assert targets.shape == (2 * n, 1)
```

This adaptation uses a local random generator so other code does not silently change its random state. A fixed seed makes repeated runs of the same setup reproducible; it does not promise matching random numbers across different libraries.

`cov=[[1, 0.5], [0.5, 1]]` is a list containing two lists. NumPy interprets it as a two-by-two covariance matrix. The two entries in `mean` correspond to the two features, so each generated array has shape `(1000, 2)`.

`np.vstack((negative, positive))` passes one tuple containing two arrays. This is not the same argument structure as passing two separate positional arguments. Stacking produces `(2000, 2)`. `.astype("float32")` is then called on that result. Read chained calls from left to right: build the stacked array, convert its numeric type, bind the result to `inputs`.

Labels have shape `(2000, 1)`, matching the eventual predictions. If you accidentally use `(2000,)`, subtracting it from `(2000, 1)` can broadcast into `(2000, 2000)`. Your code may run while computing the wrong loss. Shape assertions catch a more important problem than a missing semicolon ever would.

## 21. Reading a plot as a Python expression

Source: cells 37, 43, and 44.

```python
import matplotlib.pyplot as plt

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
```

`inputs[:, 0]` selects every row and only column zero, giving shape `(2000,)`. The first two arguments are the horizontal and vertical coordinates. `c=` is a keyword understood by Matplotlib, here receiving one value per point for coloring. `plt.show()` asks the plotting system to display the figure; it does not calculate model predictions.

After training, `predictions[:, 0] > 0.5` performs an elementwise comparison. The result is a Boolean array, not one Python Boolean. Passing that array as colors groups points by predicted class. No explicit loop is needed.

In cell 44, `np.linspace(-1, 4, 100)` produces 100 evenly spaced coordinates including the endpoints. The string `"-r"` is a plot-format argument for a red line. It is not a subtraction expression because it is inside quotes.

The boundary equation assumes the second coefficient is nonzero. Reading it as `y = (-w0 / w1) * x + (0.5 - bias) / w1` also reveals operator precedence: multiplication and division bind before addition. A vertical boundary needs separate handling if `w1` is zero. For a PyTorch plot, first use a detached CPU array such as `predictions.detach().cpu().numpy()`; plotting libraries should not receive a tensor still participating in gradient tracking.

## 22. TensorFlow: names, objects, and mutation

Source: “Tensor assignment and the Variable class,” cells 18–22.

```python
import tensorflow as tf

v = tf.Variable(tf.zeros((3, 1)))
alias = v
v.assign(tf.ones((3, 1)))
v[0, 0].assign(3.0)
v.assign_add(tf.ones((3, 1)))
```

`alias = v` does not copy the variable. Both names refer to the same object. `assign` changes that object's stored value, so reading either name sees the update. The final values here are `[[4.], [2.], [2.]]`.

Now compare `v = tf.ones((3, 1))`. This rebinds the name `v` to a tensor. It does not update the previous variable object; `alias` still refers to that object. A Python variable name and a TensorFlow `Variable` are different concepts. Every Python name can be rebound, but a TensorFlow variable additionally supports stored numerical state that assignment methods can change.

The indexed `v[0, 0].assign(...)` form in the example uses TensorFlow's supported variable-slice assignment. Ordinary constant tensors do not support NumPy-style item mutation such as `x[0, 0] = 3`.

This distinction explains training code. `W.assign_sub(change)` modifies the same weight object on each step. `W = W - change` merely binds a local name to a result and can lose the variable behavior the training code expects.

## 23. `with ... as ...` controls a region of execution

Source: “Gradients in TensorFlow,” cells 27–29.

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    result = tf.square(x)
gradient = tape.gradient(result, x)
```

`with` is Python's context-management statement. It enters a context, runs the indented block, and exits that context even when an exception occurs. `as tape` binds the value returned on entry to the name `tape`. This particular context records eligible tensor operations for differentiation.

Notice where indentation ends. Computing `result` is inside recording; requesting `gradient` is outside. A `with` block does not create a new Python local scope, so both names remain accessible afterward. Moving `result = ...` above the block changes what is recorded even though every name still exists.

A trainable `tf.Variable` accessed inside the tape is watched automatically. A constant needs `tape.watch(x)` before its recorded computation. The default nonpersistent tape normally supports one gradient request. These are TensorFlow rules layered on top of Python's context syntax. [TensorFlow GradientTape reference](https://www.tensorflow.org/api_docs/python/tf/GradientTape).

### Why nested contexts matter

Notebook excerpt, cell 29:

```python
time = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time**2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
```

The inner context finishes before `speed` is computed, but the outer context is still active. That placement lets the outer tape record the derivative calculation. `**` means exponentiation; the first derivative is `9.8 * time`, and its derivative is `9.8`. At `time = 0`, speed is zero without making acceleration zero. Moving the speed calculation outside both contexts would destroy this arrangement.

## 24. A complete TensorFlow step, read as Python

Source: linear-classifier cells 38–42. Dependencies: TensorFlow and the NumPy `inputs` and `targets` from section 20. The following two blocks belong together.

```python
W = tf.Variable(tf.random.uniform((2, 1)))
b = tf.Variable(tf.zeros((1,)))
learning_rate = 0.1

def linear_tf(inputs, W, b):
    return tf.matmul(inputs, W) + b

def mse_tf(targets, predictions):
    return tf.reduce_mean(tf.square(targets - predictions))
```

```python
def step_tf(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = linear_tf(inputs, W, b)
        loss = mse_tf(targets, predictions)
    grad_W, grad_b = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * grad_W)
    b.assign_sub(learning_rate * grad_b)
    return loss

for step in range(40):
    loss = step_tf(inputs, targets, W, b)
    print(f"Step {step}: {float(loss):.4f}")
```

`W` and `b` are TensorFlow variables; `predictions` is a tensor of shape `(2000, 1)`; `loss` is a scalar tensor with shape `()`. TensorFlow converts these NumPy inputs when its operations consume them. The weights have shape `(2, 1)` and the bias has shape `(1,)`, so the bias broadcasts across all rows.

`[W, b]` groups the two differentiation targets in order. The return value contains corresponding gradients. `grad_W, grad_b = ...` unpacks it; Python requires exactly two elements. An unconnected gradient can be `None`, so a later multiplication error may actually indicate a broken dependency earlier in the computation.

The learning rate is read from surrounding module scope. The function does not assign to that name, so it needs no `global` statement. Parameters `W` and `b` are local names referring to the caller's variable objects. Mutation explains why the caller sees updated weights even though only `loss` is returned.

The notebook's cell 41 passes `predictions, targets` to a function declared as `mean_squared_error(targets, predictions)`. Squaring makes this particular loss symmetric, so the number is unchanged. Preserve the intended argument order anyway; other losses are not symmetric.

## 25. Decorators: passing a function instead of calling it

Source: TensorFlow cells 31–32, PyTorch 78–79, JAX 112 and 117.

```python
@tf.function
def twice(x):
    return 2 * x
```

Conceptually, Python creates the function and then performs `twice = tf.function(twice)`. The decorator receives the function object. `twice(3)` would instead call the function and produce a value. Omitting parentheses can be exactly the right thing when an API expects behavior to wrap or transform.

For `@tf.function(jit_compile=True)`, Python first calls `tf.function` with the configuration. The resulting decorator receives the function. At the language level, a decorated definition is convenient spelling for function transformation and rebinding.

The `@` in `a @ b` means matrix multiplication. Its position between expressions distinguishes it from a decorator line above `def`. The character is shared; the operations are different.

`torch.compile(model)` similarly receives an existing callable object and returns a compiled callable. A model instance can be callable without being a plain function. `@jax.jit` decorates a function using the same Python syntax.

Compilation is optional for learning these steps. TensorFlow and JAX may trace Python functions and reuse compiled computations, so Python side effects such as `print` or mutation of an ordinary list are not reliable per-step logging mechanisms inside traced code. Keep simple reporting outside and establish a working uncompiled version first. Compilation support also depends on the operations and environment. [TensorFlow function guide](https://www.tensorflow.org/guide/function), [JAX JIT guide](https://docs.jax.dev/en/latest/jit-compilation.html).

## 26. PyTorch: attributes that store gradients

Source: “Computing gradients with PyTorch,” cells 63–65.

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
torch.square(x).backward()
print(x.grad)                    # tensor(6.)
torch.square(x).backward()
print(x.grad)                    # tensor(12.)
x.grad = None
```

`requires_grad=True` configures this tensor for gradient tracking. `backward()` is a method call on the result tensor. It has a side effect: relevant leaf tensors accumulate gradient values in `.grad`. The method does not return the gradient for you to assign.

The second square is a new forward computation. Its gradient adds to the previous one, producing twelve. This differs from trying to backpropagate through the same already-freed graph twice. Explicit accumulation can be useful, but this chapter's training loop wants fresh gradients each step. [PyTorch backward reference](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html).

`None` is Python's singleton for absence, not numeric zero. `x.grad = None` assigns an attribute, clearing the stored gradient. `x = None` would instead discard the local binding to the tensor. `x.grad == None` would be a comparison, not a reset. Use `x.grad is None` when checking absence.

The notebook's dense helpers in cells 61 and 79 use `torch.nn.relu`, which is not the functional API. Use `torch.nn.functional.relu(...)` or `torch.relu(...)`; the module class is `torch.nn.ReLU`. Capitalization and namespaces are part of an API's contract. [PyTorch functional ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html).

## 27. A corrected PyTorch step with explicit state

Source: “An end-to-end example: A linear classifier in pure PyTorch,” cells 67–70. Dependencies: section 20 data and `import torch`. The function name is deliberately distinct from later model objects.

```python
torch_inputs = torch.tensor(inputs, dtype=torch.float32)
torch_targets = torch.tensor(targets, dtype=torch.float32)
W = torch.rand(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
learning_rate = 0.1

def linear_torch(inputs, W, b):
    return inputs @ W + b

def mse_torch(targets, predictions):
    return torch.mean(torch.square(targets - predictions))
```

```python
def step_torch(inputs, targets, W, b):
    predictions = linear_torch(inputs, W, b)
    loss = mse_torch(targets, predictions)
    loss.backward()
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    W.grad = None
    b.grad = None
    return loss

for step in range(40):
    loss = step_torch(torch_inputs, torch_targets, W, b)
    print(f"Step {step}: {loss.item():.4f}")
```

Cell 70 calls `model(inputs)` even though cell 68 defines `model(inputs, W, b)`. In that sequence Python raises `TypeError` for missing arguments. The adaptation supplies all three arguments and converts the shared NumPy arrays explicitly to PyTorch tensors.

For these tensors, `-=` performs an in-place update. It preserves the weight objects used by the caller. This statement is inside `torch.no_grad()` so the parameter update itself is not added to the reverse-mode gradient computation. The context has no `as` clause because we do not need a context-bound name. [PyTorch no_grad reference](https://docs.pytorch.org/docs/stable/generated/torch.no_grad).

After leaving the context, gradient tracking resumes for ordinary operations. Clearing gradients belongs after their use or before the next backward pass, never between `backward()` and the update. `.item()` extracts a Python scalar from a one-element tensor for formatting. Do not replace the tensor loss by that scalar before `backward()`; Python floats have no gradient history.

## 28. Inheritance packages state with behavior

Source: “Packaging state and computation with the Module class,” cells 72–76.

```python
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.rand(input_dim, output_dim)
        )
        self.b = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        return inputs @ self.W + self.b

torch_model = LinearModel()
optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)
```

The parentheses in `class LinearModel(torch.nn.Module)` declare inheritance. `super().__init__()` initializes the inherited module machinery before registered parameters are assigned. `self` is a conventional parameter name for the current instance; Python supplies that instance when you call a bound method.

`self.W` persists as an attribute between calls. A bare local `W` inside `__init__` would disappear as a local binding when initialization finishes unless something retained it. Assigning a `Parameter` to a module attribute allows the module to discover it. `parameters()` is a method returning an iterator over registered parameters; the optimizer consumes those objects. It does not receive the method object because the parentheses are present.

Call `torch_model(torch_inputs)`, rather than directly calling `.forward(torch_inputs)`. Here `torch_inputs` is the PyTorch tensor converted in the preceding section; the shared `inputs` is still a NumPy array. The inherited `__call__` behavior routes through module machinery and then invokes your `forward`. Python's callable-object protocol explains the function-like spelling. Keras uses the same outward spelling but a different customization method named `call`.

```python
def step_module(inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    predictions = torch_model(inputs)
    loss = mse_torch(targets, predictions)
    loss.backward()
    optimizer.step()
    return loss
```

This version reads the model and optimizer from surrounding scope. It also uses `mse_torch` from the preceding section. Call it with `loss = step_module(torch_inputs, torch_targets)`. Moving the model and optimizer into arguments would make dependencies more explicit. The notebook resets with `model.zero_grad()` afterward; either placement can work when gradients are cleared between steps. Do not apply the manual update as well as `optimizer.step()` unless you intentionally want two updates.

## 29. JAX: familiar syntax, immutable arrays

Source: “Tensors in JAX” and “Tensor assignment,” cells 84–86 and 98–100.

```python
import jax
from jax import numpy as jnp

x = jnp.array([1, 2, 3], dtype="float32")
old_x = x
new_x = x.at[0].set(10)
x = x * 2
```

`from jax import numpy as jnp` imports JAX's NumPy-like namespace and gives it a short local name. It does not turn ordinary NumPy into JAX. The functions look familiar, but their returned objects have different behavior.

`x.at[0].set(10)` returns an updated array, leaving the original array unchanged. Read the chain as attribute access, indexing of the update helper, then a method call. Afterward, `old_x` still denotes `[1, 2, 3]`, `new_x` denotes `[10, 2, 3]`, and rebound `x` denotes `[2, 4, 6]`. [JAX indexed-update reference](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

Cell 100 uses `e *= d`. For immutable JAX arrays, this produces a result and rebinds `e`; it does not mutate the array previously referred to by `e`. Never infer mutability solely from `+=` or `*=`. Python lets an object's type define augmented-assignment behavior.

This is why the JAX training step must return updated weights and why the caller must capture them. TensorFlow variables and PyTorch parameters in the earlier examples support object mutation; JAX's array-based example communicates state through returned values.

## 30. Random state can be implicit or explicit

Source: “Random number generation in JAX,” cells 88–96.

The notebook first demonstrates NumPy's global random generator. Two calls consume successive state, while resetting the seed makes a sequence repeat. Its cell 90 contains an argument-binding mistake: `np.random.normal((3,))` passes `(3,)` as `loc`, the distribution's mean. The requested three-element output needs `size=(3,)`. [NumPy normal signature](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).

An adaptation that states both the input shape and random dependency is:

```python
def apply_noise(x, seed):
    rng = np.random.default_rng(seed)
    return x * rng.normal(size=x.shape)

x = np.ones(3, dtype="float32")
y = apply_noise(x, 1337)
z = apply_noise(x, 1338)
```

Here `seed` is a parameter, and `rng` is a local object. The function leaves `x` unchanged because it returns multiplication's result. Passing the same seed and input repeats the result. This uses multiplication by noise, not addition of noise.

JAX makes a random key an explicit input:

```python
key = jax.random.key(1337)
key, draw_key = jax.random.split(key)
noise = jax.random.normal(draw_key, shape=(3,))
key, next_draw_key = jax.random.split(key)
next_noise = jax.random.normal(next_draw_key, shape=(3,))
```

Each split yields new keys. Unpacking keeps one for future splitting and uses another for the draw. The right-hand side uses the old value of `key` before the left-hand assignments happen. Reusing an identical key with an identical sampling call repeats its result; drawing does not automatically advance a hidden generator. [JAX pseudorandom-number guide](https://docs.jax.dev/en/latest/random-numbers.html).

Cell 96 uses `split(..., num=1)[0]`: get a collection containing one new key, then retrieve its first element. The two-key pattern above makes ongoing state management clearer. A seed integer and a JAX key object are not interchangeable merely because both relate to randomness.

## 31. Functions can accept and return functions

Source: “Computing gradients with JAX,” cells 103–108, and classifier cells 114–117.

```python
def square_loss(x):
    return jnp.square(x)

grad_fn = jax.grad(square_loss)
gradient = grad_fn(jnp.array(3.0))
value_and_grad_fn = jax.value_and_grad(square_loss)
value, gradient = value_and_grad_fn(jnp.array(3.0))
```

At the first assignment, `square_loss` is passed without parentheses. JAX receives the function itself and creates another function. The next line calls that derivative function on a value. The results at three are value nine and gradient six. Function names are ordinary bindings: they can be passed, assigned, and replaced just like other objects.

For `value_and_grad`, the differentiated function must return a scalar. An array of shape `(1,)` is still a vector, even if it has one element. `jnp.mean(...)` in the classifier creates the required scalar. By default the derivative is with respect to the first positional argument; `argnums` selects others. A tuple of weight arrays can occupy that single argument. [JAX value_and_grad reference](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html).

### Tuple state and nested return values

```python
def loss_jax(state, inputs, targets):
    W, b = state
    predictions = inputs @ W + b
    return jnp.mean(jnp.square(targets - predictions))

loss_and_grads_jax = jax.value_and_grad(loss_jax)
```

`state` is a tuple holding two arrays. Unpacking it makes convenient local names without copying either array. The resulting gradients mirror that tuple, so calling `loss_and_grads_jax((W, b), inputs, targets)` returns `(loss, (grad_W, grad_b))`. Its distinct name avoids overwriting the earlier scalar derivative function when notebook cells are rerun.

For a diagnostic extension, return `loss, predictions` and construct the transformation with `has_aux=True`. Then unpack `(loss, predictions), grads = transformed(...)`. This flag tells JAX which output is the scalar objective; it does not ask JAX to differentiate the predictions as a second loss.

## 32. A complete JAX step returns its new state

Source: classifier cells 117–118. Dependencies: section 20 data, JAX imports, and `loss_jax`/`loss_and_grads_jax` from section 31.

```python
jax_inputs = jnp.asarray(inputs)
jax_targets = jnp.asarray(targets)
key = jax.random.key(1337)
W = jax.random.uniform(key, shape=(2, 1))
b = jnp.zeros((1,), dtype="float32")
learning_rate = 0.1

@jax.jit
def step_jax(inputs, targets, W, b):
    loss, grads = loss_and_grads_jax((W, b), inputs, targets)
    grad_W, grad_b = grads
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b
    return loss, W, b

for step in range(40):
    loss, W, b = step_jax(jax_inputs, jax_targets, W, b)
    print(f"Step {step}: {float(loss):.4f}")
```

Every update binds local names to new array values. `return loss, W, b` returns a three-element tuple even without explicit outer parentheses. The assignment in the loop unpacks that tuple, updating the caller's bindings. If you instead write only `loss = step_jax(...)`, `loss` becomes the entire tuple and the caller's `W` and `b` remain unchanged.

The notebook creates `state = (W, b)` before its loop but then passes `W` and `b` directly. That old tuple does not automatically refresh when the names are rebound. Either reconstruct a tuple when needed, as the function does, or make the training API accept and return `state` consistently.

The setup here samples JAX weights directly and explicitly converts inputs. Cell 118 instead builds arrays from NumPy draws; those values are valid inputs, but JAX dtype defaults can affect conversion. Explicit floating dtypes make teaching examples easier to inspect.

## 33. Keras: a callable class is a learning building block

Source: “Picking a backend framework,” cell 123; “The base Layer class,” cells 126–127.

In a fresh process, set `os.environ["KERAS_BACKEND"]` before importing Keras. That dictionary-like assignment configures an environment variable; it is not a general Python switch that converts existing tensors. This course's Keras examples can use the notebook's JAX backend.

In this fresh process, rerun the shared dataset block in section 20 as well. It defines `inputs`, `targets`, and the NumPy generator `rng` used in the validation-split example. Each other fresh framework environment likewise needs that data setup before its classifier examples.

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```

The following corrected adaptation is a complete class:

```python
class SimpleDense(keras.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units),
            initializer="random_normal",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros"
        )
    def call(self, inputs):
        y = keras.ops.matmul(inputs, self.W) + self.b
        return self.activation(y)
```

`__init__` stores configuration. `build` creates shape-dependent weights. `call` describes the numerical computation. The inherited `__call__` machinery coordinates building on first use and then invokes `call`; use `layer(inputs)` to enter that machinery. `add_weight` creates weights registered with the layer, while ordinary local arrays would not provide that same registration. [Keras subclassing guide](https://keras.io/guides/making_new_layers_and_models_via_subclassing/).

The notebook unpacks `batch_dim, input_dim = input_shape`, assuming a rank-two input. `input_shape[-1]` selects the final dimension directly and avoids an unused batch variable. For input `(2, 784)` and 32 units, `W` is `(784, 32)`, `b` is `(32,)`, and the output is `(2, 32)`.

### An activation string is not a function

Cell 126 stores `activation` directly, then calls it. Cell 127 passes the callable `keras.ops.relu`, so this works. Cell 131 passes `"relu"` and `"softmax"` strings to that custom class. A string cannot be called; using the layer will fail.

The corrected class resolves the identifier with `keras.activations.get`. This accepts supported string identifiers or callables; `None` resolves to the identity activation. That is why the adapted `call` no longer needs the notebook's `if self.activation is not None:` guard. [Keras activation resolver source](https://github.com/keras-team/keras/blob/master/keras/src/activations/__init__.py).

```python
layer = SimpleDense(32, activation="relu")
x = keras.ops.ones((2, 784))
y = layer(x)
print(y.shape)                         # (2, 32)
```

The expression `keras.ops.relu` retrieves a function; `keras.ops.relu(x)` calls it. The expression `"relu"` creates text. A library can choose to translate that text into a callable, but Python does not do so automatically.

## 34. Lists of objects and training configuration

Source: “Automatic shape inference,” cells 129–131, and “The compile step,” cells 134–135.

```python
keras_model = keras.Sequential([
    SimpleDense(32, activation="relu"),
    SimpleDense(10, activation="softmax"),
])
```

Square brackets build a list of two already-created layer objects. Parentheses outside the brackets pass that list to the `Sequential` constructor. Its ordering determines computation order. The second layer infers its input width from the first layer's output.

For the shared two-feature, binary-label dataset, use the notebook's one-output model instead:

```python
keras_model = keras.Sequential([keras.layers.Dense(1)])
keras_model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
```

Each constructor call creates an object. `metrics=[...]` supplies a list because several metrics may be requested. The string alternatives in cell 134 are identifiers that Keras resolves, just as built-in activation arguments are resolved. Passing an instantiated optimizer gives you a place to configure `learning_rate` explicitly.

This particular `compile` method configures training: optimizer, loss, metrics, and execution settings. It is not the same API as `torch.compile`. Training starts with `fit`. A linear output with MSE reproduces the chapter's teaching example; it is not a guarantee that its outputs are probabilities. [Keras model training APIs](https://keras.io/api/models/model_training_apis/).

## 35. Shuffle paired data with one permutation

Source: “Monitoring loss and metrics on validation data,” cell 141.

```python
permutation = rng.permutation(len(inputs))
shuffled_inputs = inputs[permutation]
shuffled_targets = targets[permutation]

n_val = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:n_val]
val_targets = shuffled_targets[:n_val]
train_inputs = shuffled_inputs[n_val:]
train_targets = shuffled_targets[n_val:]
```

`len(inputs)` returns its first-axis length, here 2000. The permutation is an integer array containing every row index once. Indexing with that array performs advanced indexing: select rows in the specified order, creating new NumPy arrays. Reusing the same order for targets preserves each example-label pair. Independent shuffles silently damage the dataset.

`0.3 * len(inputs)` is a floating-point calculation. `int(...)` converts it to an integer by truncating toward zero. With a nonnegative count this discards any fractional remainder. Python slice endpoints need integer-like values, not `600.0` merely because it represents a whole number.

The slice `[:n_val]` excludes index `n_val`; `[n_val:]` starts there. This partitions the shuffled arrays without overlap. Basic NumPy slices generally produce views, so avoid modifying these subsets in place if you need the backing shuffled arrays preserved. Validation has 600 samples and training has 1400 here.

Shuffling before splitting matters because the original dataset puts all zero labels before all one labels. This random split is appropriate for this synthetic example; it is not a universal instruction to shuffle time-dependent datasets.

## 36. Return objects, dictionaries, and prediction batches

Source: “Understanding the fit method,” cells 138–139, and inference cell 143.

```python
history = keras_model.fit(
    train_inputs, train_targets,
    epochs=5, batch_size=16,
    validation_data=(val_inputs, val_targets),
)
losses = history.history["loss"]
last_loss = losses[-1]
predictions = keras_model.predict(val_inputs, batch_size=128)
print(predictions[:10])
```

`validation_data=(val_inputs, val_targets)` is one keyword argument whose value is a pair. It does not pass two independent validation arguments. `epochs=5` repeats training over the dataset five times; `batch_size=16` groups training examples for each step. Prediction batching controls how examples are processed, not how many predictions are ultimately returned.

The left `history` is your variable name. The right `.history` is an attribute on the returned History object. It holds a dictionary mapping names to lists of epoch results. `history.history["loss"]` first retrieves the dictionary, then looks up a string key. `[-1]` retrieves the final list element. A missing dictionary key raises `KeyError`; it does not return an empty list.

Inspect actual metric keys instead of guessing:

```python
for name, values in history.history.items():
    print(f"{name}: final value = {values[-1]:.4f}")
```

`.items()` returns key-value pairs that the loop unpacks into two names. This pattern also works for ordinary dictionaries unrelated to machine learning. The result of `predict` for the one-output model has shape `(600, 1)`; `predictions[:10]` keeps the first ten rows and the column dimension.

## 37. Exercises: make the state changes explicit

Attempt these before reading section 38. Explain each answer in words as well as code.

### Exercise A — Repair three calls

Explain and repair `np.random.normal((3,))`, `torch.nn.relu(x)`, and `linear_torch(inputs)` when the function definition requires `inputs, W, b`. For the last call, also identify the tensor-type requirement if `inputs` came from NumPy.

### Exercise B — Follow aliases

For TensorFlow, create `v = tf.Variable(1.0)` and `old = v`, then call `v.assign_add(2.0)` and finally assign `v = tf.constant(9.0)`. Predict the value and object kind referred to by each name. Repeat the reasoning for JAX with `x = jnp.array([1.0])`, `old = x`, and `x += 2`.

### Exercise C — Catch an accidental outer difference

You have predictions with shape `(4, 1)` and targets with shape `(4,)`. Predict the shape of `targets - predictions`. Write a small NumPy example that demonstrates it, then repair the targets without changing which sample each value belongs to.

### Exercise D — Debug gradient accumulation

For `x = torch.tensor(2.0, requires_grad=True)`, run a newly computed `x**2` backward three times without clearing gradients. Predict `.grad` after each run. Then write a loop in which each iteration prints only that iteration's gradient.

### Exercise E — Refactor a JAX step

Rewrite `step_jax` to accept `(state, inputs, targets)` and return `(loss, new_state)`. Show the calling loop. Explain why creating `state = (W, b)` once does not keep it synchronized with future rebindings of `W` and `b`.

### Exercise F — Verify paired shuffling

Make five input rows containing IDs zero through four. Define each target as ten times its row's ID. Shuffle both arrays using one permutation, split off two validation rows, and assert the target relationship still holds in both subsets.

### Exercise G — Make an activation contract visible

Create the corrected `SimpleDense` with `activation=None`, `activation="relu"`, and a callable activation. Explain when each activation is resolved, when weights are created, and which method actually implements the tensor arithmetic. Predict the weight and output shapes for four input rows, three features, and five units.

### Exercise H — Read a training history

Given `logs = {"loss": [0.8, 0.4, 0.3], "val_loss": [0.9, 0.5, 0.6]}`, print the epoch with the lowest validation loss using one-based numbering. Explain why the last training loss and the lowest validation loss need not occur at the same epoch.

## 38. Worked solutions and reasoning

### Solution A — Argument binding precedes intention

Use `np.random.normal(size=(3,))`, `torch.nn.functional.relu(x)`, and `linear_torch(torch_inputs, W, b)`. Build `torch_inputs = torch.tensor(inputs, dtype=torch.float32)` if needed. Python binds positional arguments by position; it cannot infer that `(3,)` was intended as a size. Attribute lookup must find the named function before a call can occur. A function with three required parameters needs all three supplied.

### Solution B — Names are not storage containers

After TensorFlow mutation, `old` and `v` refer to the same variable containing `3.0`. After rebinding, `v` refers to a constant tensor containing `9.0`, while `old` still refers to the variable containing `3.0`. In JAX, `old` remains `[1.0]`; `x` is rebound to an array containing `[3.0]`. The augmented assignment cannot mutate an immutable array.

### Solution C — Shape alignment can conceal a bug

```python
p = np.array([[10.], [20.], [30.], [40.]])
t = np.array([1., 2., 3., 4.])
assert (t - p).shape == (4, 4)
t = t.reshape(-1, 1)
assert (t - p).shape == (4, 1)
```

The original target is treated as a row during broadcasting, so every prediction is compared with every target. `reshape(-1, 1)` lets NumPy infer the row count and preserves element order. A working subtraction is not proof of a meaningful subtraction.

### Solution D — Clear before the next accumulation

The successive gradients are four, eight, and twelve. Each fresh computation contributes another `2*x = 4`.

```python
x = torch.tensor(2.0, requires_grad=True)
for _ in range(3):
    x.grad = None
    loss = x**2
    loss.backward()
    print(x.grad.item())              # 4.0 every time
```

The underscore is an ordinary variable name conventionally indicating an unused loop value. No parameter update occurs in this exercise, so the derivative's input stays two.

### Solution E — Return the container you want to preserve

```python
@jax.jit
def step_state(state, inputs, targets):
    loss, grads = loss_and_grads_jax(state, inputs, targets)
    W, b = state
    grad_W, grad_b = grads
    new_state = (
        W - learning_rate * grad_W,
        b - learning_rate * grad_b,
    )
    return loss, new_state

state = (W, b)
for _ in range(40):
    loss, state = step_state(state, jax_inputs, jax_targets)
```

A tuple holds references to the objects supplied when it is constructed. It does not hold live references to the future meanings of variable names. Capturing `new_state` makes state propagation explicit and prevents accidentally training from the original weights repeatedly.

### Solution F — One index order protects the relationship

```python
rows = np.arange(5).reshape(-1, 1)
labels = 10 * rows
order = np.random.default_rng(7).permutation(len(rows))
rows, labels = rows[order], labels[order]
val_x, val_y = rows[:2], labels[:2]
train_x, train_y = rows[2:], labels[2:]
assert np.array_equal(val_y, 10 * val_x)
assert np.array_equal(train_y, 10 * train_x)
```

The relationship test is stronger than merely checking equal lengths. Independent permutations would usually preserve length while corrupting pairings. Both right-hand indexing expressions are evaluated before either left-hand binding changes.

### Solution G — Initialization, building, and calling differ

The constructor resolves the activation using `keras.activations.get`. `None` becomes an identity callable; the string identifies a supported activation; a supplied callable already represents behavior. First calling the layer with inputs triggers weight building. The layer's `call` method performs matrix multiplication, bias addition, and activation. For `(4, 3)` inputs and five units, weights are `(3, 5)`, bias is `(5,)`, and output is `(4, 5)`.

```python
for activation in (None, "relu", keras.ops.relu):
    layer = SimpleDense(5, activation=activation)
    result = layer(keras.ops.ones((4, 3)))
    assert tuple(result.shape) == (4, 5)
```

### Solution H — The best item can be located by its value

```python
logs = {
    "loss": [0.8, 0.4, 0.3],
    "val_loss": [0.9, 0.5, 0.6],
}
val_losses = logs["val_loss"]
best_index = min(range(len(val_losses)), key=val_losses.__getitem__)
print(best_index + 1)                 # 2
```

`range` supplies candidate indices. `key=` receives a callable that tells `min` how to score each index. `val_losses.__getitem__` is the list's bound indexing method; passing it does not call it immediately. An easier equivalent is `best_index = val_losses.index(min(val_losses))`, which chooses the first occurrence if the minimum is tied. Training loss and validation loss measure different data, so one can decrease while the other rises.
