# Similarity Measures Repository

<!-- ![Backend metrics](figures/backend_metrics.png) -->
![Backend metrics](https://github.com/nacloos/similarity-measures/blob/main/figures/backend_metrics.png)
    
Aggregate existing implementations of similarity measures into a single python package.

No measure is re-implemented here! Leverage only existing implementations.


## Installation

 ```
 pip clone https://github.com/nacloos/similarity-measures.git
 cd similarity-measures
 pip install -e .
 ```

TODO: does it work?
```
pip install git+https://github.com/nacloos/similarity-measures.git
```

## Usage

```python
import similarity

metric = similarity.make("cca")
```

## Standardized metric interface
### How do I create a metric?
```python
import similarity
metric = similarity.make(metric_id)
```
But what is that metric id thing?

### What can I do with a metric?
Just `print(metric)` and you will get a description of what you can do.

```python
print(metric)
# TODO: add example output

metric.fit(X, Y)
metric.score(X, Y) -> float
     # rationale: it can be useful to fit and evaluate on different data for example, when using cross-validation
metric.fit_score(X: np.ndarray[sample, neuron], Y: np.ndarray[sample, neuron]) -> float
    # rationale: a quick way to fit and evaluate the metric on the same data
```

If you want a more in-depth description, you can print each method individually:
```python
print(metric.fit_score)

```



### Why this particular interface?
sklearn

### How can I change the interface?
If you want to change it for your own usage, just specify the interface you want when creating the metric. 
```python
metric = similarity.make(
    metric_id, 
    interface={
        "fit_score": "__call__"  # metric can now be used as a function
    }
)
...
score = metric(X, Y)
```

If you want to have your own custom metric creator, you can leverage python functools' `partial` function:
```python
from functools import partial

make_metric = partial(similarity.make, interface={"fit_score": "__call__"})
metric = make_metric(metric_id)
...
score = metric(X, Y)
```


If you want to suggest modifications to the standard interface, please open an issue.

## Organization of the repository

## Why use CUE instead of plain python?
Can easily generate a json config describing the config

Why cue language? Can use schema to validate config. Show example of adding a metric that doesn't have a card

## Adding an implementation of an existing metric
* create a folder in `similarity/backend`
* create a `requirements.txt` file with the dependencies of the backend. Optionally add a comment with the link to the installation instructions (e.g. in the README of the backend).


### Adding a new metric
(Or "Registering a new implementation")

Can only add an metric implementation for which there exists a card. Otherwise, create a card first.

Have to only modify the backend folder to add a new backend.
Create a folder for your backend. Create a cue file with the backend package.
Add an import for your backend and add the id/name of the backend in `backends.cue`


Import backend in `similarity/backend/backends.cue`
Add a new line in the import statement:
```
{backend_id} "github.com/similarity/backend/{backend_folder}:backend"
```
Add an entry to `#backends`

Checklist:
* is your metric importable from a python package?
  * yes: add the package is a requirement
  * no: add the code to the backend folder
* is the metric implemented as a class or a function?
* what are the expected arguments? Data type and shape?
  * what transformation is need to go from the standard input to the backend input? See the [Metric interface](#standardized-metric-interface).
  * if just need to rename the inputs, use #fit_score_inputs
  * if need to transform the inputs, use #preprocessing



### Adding a new benchmark
Either copy paste code
* link to commit from which the code was copied
or put the code in a python package and link to it


### Adding an new implementation of an existing metric

