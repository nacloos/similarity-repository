# Similarity Repository


![Implemented measures](https://github.com/nacloos/similarity-repository/blob/main/figures/measures.png)


A unified Python package that standardizes existing implementations of similarity measures to faciliate comparisons across studies. 

**Paper:** [A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field](https://openreview.net/pdf?id=vyRAYoxUuA)


## Installation
Install via pip:
```bash
pip install git+https://github.com/nacloos/similarity-repository.git
```

For faster installation using `uv` in a virtual environment (add `--system` to install outside of virtual environment):
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

# generate two datasets
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# measure their similarity
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

similarity.register("my_repo/my_measure", my_measure)

# use it like any other measure
measure = similarity.make("my_repo/my_measure")
score = measure(X, Y)
```

## Project structure

* [`similarity/registry`](similarity/registry/): all the registered github repositories
* [`similarity/standardization.py`](similarity/standardization.py): mapping to standardize names and transformations to leverage relations between measures
* [`similarity/papers.py`](similarity/papers.py): information about papers for each github repository in the registry
* [`similarity/types/__init__.py`](similarity/types/__init__.py): list with all the registered identifiers


## Contributing
If your implementation of similarity measures is missing, please contribute!

Follow these steps to register your own similarity measures:
* Fork the repository.
* Create a new folder in [`similarity/registry/`](similarity/registry/) for your repository and a `__init__.py` file inside it.
* Register your measures using `similarity.register`. The easiest way is to copy your code with the similarity measures into the created folder and import them in your  `__init__.py` file.
* Use the naming convention `{repo_name}/{measure_name}` (you can use any measure name under your own namespace).

* Add your folder to imports in [`similarity/registry/__init__.py`](similarity/registry/__init__.py).
* Add your paper to [`similarity/papers.py`](similarity/papers.py).

You can then check that your measures have been registered correctly:
```python
import similarity

X, Y = np.random.randn(50, 30), np.random.randn(50, 30)
measures = similarity.make("{repo_name}/{measure_name}")
score = measures(X, Y)
```

If you want to map your measures to standardized names, see [`similarity/standardization.py`](similarity/standardization.py). Standardized measures are under the `measure/` namespace and have the form `measure/{repo_name}/{standardized_measure_name}`. If your measure already exists in another repository, you can use the same standardized name. In this case, make sure your implementation is consistent with the existing ones. If your measure is new, you can propose a new standardized name.


Submit a pull request for your changes to be reviewed and merged.

For additional questions for how to contribute, please contact nacloos@mit.edu.

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
