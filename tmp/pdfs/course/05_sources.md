# Sources and verification

## What this course is based on

The primary sources are the local third-edition companion notebooks for *Deep Learning with Python* by Francois Chollet and Matthew Watson, together with the repository README and license. All 165 code cells across chapters 2 and 3 were inspected. Explanations are original; source excerpts and adaptations are identified in the lessons. Multiple repetitions of the same Python pattern are consolidated rather than printed repeatedly.

Local notebook basenames:

- `chapter02_mathematical-building-blocks.ipynb`
- `chapter03_introduction-to-ml-frameworks.ipynb`

The repository README identifies the main-folder notebooks as the third edition. Source locators refer to those local files as inspected on 26 October 2026. An upstream notebook may later change, so its cell positions or mistakes may differ.

| Source | Purpose |
| --- | --- |
| [Companion repository](https://github.com/fchollet/deep-learning-with-python-notebooks) | Code provenance and environment guidance |
| [Chapter 2 notebook](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter02_mathematical-building-blocks.ipynb) | Numerical operations, classes, manual training |
| [Chapter 3 notebook](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/chapter03_introduction-to-ml-frameworks.ipynb) | Framework syntax and state-management examples |

## Further reading for a specific question

Use these references when you need the precise contract of an unfamiliar operation. They supplement the local notebook analysis; they are not prerequisites for working through the course.

- [Python tutorial: control flow and functions](https://docs.python.org/3/tutorial/controlflow.html) - calls, parameters, loops, and definitions.
- [Python tutorial: classes](https://docs.python.org/3/tutorial/classes.html) - instances, attributes, methods, and inheritance.
- [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html) - integer indices, slices, and advanced indexing.
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - shape alignment for elementwise operations.
- [NumPy copies and views](https://numpy.org/doc/stable/user/basics.copies.html) - shared storage and explicit copies.
- [NumPy normal sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html) - positional `loc` versus keyword `size`.
- [IPython custom magics](https://ipython.readthedocs.io/en/stable/config/custommagics.html) - the backend helper's registration pattern.
- [TensorFlow GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) - watched variables, constants, and gradient requests.
- [TensorFlow function guide](https://www.tensorflow.org/guide/function) - tracing and compiled function behavior.
- [PyTorch backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) - accumulation into gradient attributes.
- [PyTorch no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad) - the update context in the manual step.
- [PyTorch functional ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html) - the correct functional activation namespace.
- [JAX random numbers](https://docs.jax.dev/en/latest/random-numbers.html) - explicit random keys and splitting.
- [JAX value_and_grad](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html) - scalar objectives, argument selection, and auxiliary outputs.
- [JAX JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html) - function transformation and tracing.
- [Keras layer subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) - `build`, `call`, and tracked weights.
- [Keras training APIs](https://keras.io/api/models/model_training_apis/) - compile, fit, evaluate, and predict.

## Verification and code license

The final lab's complete NumPy pipeline was executed, including shape, dtype, batch-boundary, original-data-preservation, prediction, and accuracy assertions. Additional deterministic Python/NumPy examples were run. Every Python code block in the finished course passed a syntax compilation check.

Small TensorFlow/Keras checks were executed locally, including the watched-constant gradient, gradients for Keras variables, optimizer gradient-variable pairing, and activation-resolution repairs. These checks used TensorFlow 2.21.0 and Keras 3.15.1. No full MNIST training run was needed to validate the Python lessons. PyTorch and JAX examples were reviewed against their APIs but were not executed because those packages were not installed in the available local environment. Random outputs and learned metrics are therefore not presented as fixed verified results.

Code from the companion repository is provided under the following license. The course's explanatory prose is newly written.

**MIT License**

Copyright (c) 2017-present François Chollet

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

