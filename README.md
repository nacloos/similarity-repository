# Similarity Repository


![Backend metrics](https://github.com/nacloos/similarity-repository/blob/main/figures/standardization/measures.png)


A unified Python package that standardizes **existing** implementations of similarity measures to faciliate comparisons across studies. This repository does not provide new implementations - it **integrates** and **standardizes** existing ones.

**Paper:** [A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field](https://openreview.net/pdf?id=vyRAYoxUuA)


## Installation
Install via pip:
```bash
pip install git+https://github.com/nacloos/similarity-repository.git
```

For faster installation using `uv`:
```bash
pip install uv
uv pip install git+https://github.com/nacloos/similarity-repository.git
```

Alternatively, clone and install locally:
```bash
git clone https://github.com/nacloos/similarity-repository.git
cd similarity-repository
pip install -e .
```

## Usage

### Quick Start
```python
import numpy as np
import similarity

# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# make a measure object
measure = similarity.make("measure/netrep/procrustes-distance=angular")
score = measure(X, Y)
```

### Measure Identifiers
Each similarity measure has a unique identifier composed of three parts:
* the object type (i.e. `measure`)
* the repository name
* the measure name


See [`similarity/types/__init__.py`](similarity/types/__init__.py) for a complete list of implemented measures.


### Standard Interface
All measures follow this interface:
- **Inputs**: `X, Y` - numpy arrays of shape `(n_samples, n_features)`
- **Output**: `score` - float value


### Working with Multiple Measures

Select all measures from a specific repository:
```python
measures = similarity.make("measure/netrep/*")
for name, measure in measures.items():
    score = measure(X, Y)
    print(f"{name}: {score}")
```

Select all implementations of a specific measure across repositories:
```python
measures = similarity.make("measure/*/procrustes-distance=angular")
for name, measure in measures.items():
    score = measure(X, Y)
    print(f"{name}: {score}")
```

### Custom Measures

Register your own measure:
```python
# register the function with a unique id
def my_measure(x, y):
    return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))

similarity.register("measure/my_repo/my_measure", my_measure)

# use it like any other measure
measure = similarity.make("measure/my_repo/my_measure")
score = measure(X, Y)
```

## Project structure

* [`similarity/registry`](similarity/registry/): registers github repositories
* [`similarity/standardization.py`](similarity/standardization.py): standardizes measure names and applies transformations to derive new measures
* [`similarity/papers.py`](similarity/papers.py): contains the papers referenced in the repository


## Contributing
Contributions are welcome! Follow these steps:
* Create a new folder in [`similarity/registry/`](similarity/registry/) for your repository and a `__init__.py` file inside it
* Register your measures using `similarity.register`. The easiest way is to copy your code with the similarity measures into the created folder, so that you can import and register them in your  `__init__.py` file
* Add your folder to imports in [`similarity/registry/__init__.py`](similarity/registry/__init__.py)
* Add your paper to [`similarity/papers.py`](similarity/papers.py)
* Submit a pull request


 ## Citation

 ```bibtex
 @inproceedings{
    cloos2024framework,
    title={A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field},
    author={Nathan Cloos and Guangyu Robert Yang and Christopher J Cueva},
    booktitle={UniReps: 2nd Edition of the Workshop on Unifying Representations in Neural Models},
    year={2024},
    url={https://openreview.net/forum?id=vyRAYoxUuA}
}
```


## Contact

For questions or feedback, please contact:
- Nathan Cloos (nacloos@mit.edu)
- Christopher Cueva (ccueva@gmail.com)
