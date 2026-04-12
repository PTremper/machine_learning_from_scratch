# Machine Learning from Scratch

This repository contains a collection of machine learning algorithms implemented from scratch in pure Python, using only minimal external dependencies (primarily NumPy).

The goal of this project is not performance or production-readiness, but to develop a deeper understanding of how these algorithms work internally by building them from first principles.

---

## Motivation

In most practical settings, machine learning is done using well-established libraries. While this is efficient, it can obscure the underlying mechanics of the algorithms.

This project was created as a personal learning exercise to:

* Understand the core logic behind common ML methods
* Translate mathematical concepts into working code
* Explore implementation details that are usually abstracted away

---

## Implemented Algorithms

Each algorithm is implemented in its own module with minimal abstraction and a focus on clarity.

### Supervised Learning

#### Classification Models

* Decision Tree
* Random Forest
* Support Vector Machine

#### Neural Networks
* Feedforward Neural Network (Multilayer Perceptron)

#### Statistical Models

* Gaussian Process Regression

### Reinforcement Learning

* Q-Learning

*(This list may grow over time as more methods are added.)*

---

## Project Structure

Each algorithm lives in its own folder:

```
.
├── decision_tree/
│   ├── decision_tree.py
│   └── demo.py
│
├── gaussian_process_regression/
│   ├── gaussian_process_regression.py
│   └── demo.py
│
├── neural_network/
│   ├── neural_network.py
│   └── demo.py
│
├── q_learning/
│   ├── q_learning.py
│   └── demo.py
│
├── random_forest/
│   ├── random_forest.py
│   └── demo.py
│
├── support_vector_machine/
│   ├── support_vector_machine.py
│   └── demo.py
```

* Core implementations are kept as simple and self-contained as possible
* Demo scripts illustrate basic usage on toy problems

---

## Design Principles

* **Clarity over optimization**
  Code is written to be readable and understandable, not necessarily fast

* **Minimal dependencies**
  Most implementations rely only on NumPy

* **Explicit over implicit**
  Key steps (e.g. gradient calculations, tree splits) are written out rather than hidden behind abstractions

---

## Usage

Clone the repository and run any of the demo scripts:

Run `uv sync` to install required libraries into the venv. (only NumPy and tqdm for the actual code. All other libraries (like matplotlib or scikit-learn datasets) are for the demo scripts.)

Then run any demo script like

```
uv run -m machine_learning_from_scratch.random_forest.demo
```

or 

```
python -m machine_learning_from_scratch.random_forest.demo
```

from the root folder. 

Each demo is intentionally simple and meant to illustrate how the algorithm works rather than provide comprehensive functionality.

---

## Notes

* These implementations are **not intended for production use**
* Numerical stability, edge cases, and performance optimizations are not the primary focus
* Some algorithms may be incomplete or simplified

---

## Future Work

Possible extensions include:

* Additional algorithms
* Better visualization of results
* More detailed explanations or derivations
* Basic benchmarking against library implementations

---

## Disclaimer

This is a personal learning project. The code reflects an exploration process and may evolve over time as understanding improves.

Some algorithms are taken from tutorials with the links provided in the script's docstring.
