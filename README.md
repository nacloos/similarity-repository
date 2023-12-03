# Similarity Measures Repository

<!-- ![Backend metrics](figures/backend_metrics.png) -->
![Backend metrics](https://github.com/nacloos/similarity-measures/blob/main/figures/backend_metrics.png)


The goal of this repository is to gather **existing**  implementations of similarity measures for neural networks into a **single** python package, with a **common** and **customizable** interface.


<!-- No measure is re-implemented here! Leverage only existing implementations. -->


## Installation
The python package can be installed with the following command:
```
pip install git+https://github.com/nacloos/similarity-measures.git
```
Or alternatively, you can clone the repository and run `pip install -e .` inside it.


## Getting Started
Each measure is identified by a unique id (listed [here](similarity/api/__init__.py)).
Follow a naming convention:
* ´.´ to separate levels of hierarchy
* ´-´ to specify parameter values for a familly of measures. Don't use `.` for decimal numbers as it is used to separate levels of hierarchy. Use scientific notation instead (e.g. 1e-3 instead of 0.001)



```python
import similarity

# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# make a particular measure
measure = similarity.make("measure.procrustes")
score = measure.fit_score(X, Y)
```

You can also easily loop through all the available measures.
```python	
# returns a dictionary with all the measures
measures = similarity.make("measure")
for name, measure in measures.items():
    score = measure.fit_score(X, Y)
    print(f"{name}: {score}")
```

It is possible to get the configs without instantiating them into python objects using `return_config=True`. This can be useful to filter measures based on their properties. For example, to get all the measures that are scoring measures (i.e. measure of similarity where 1 is perferct similarity):
```python
score_measures = {
  k: similarity.make(f"measure.{k}")
  for k, cfg in similarity.make("measure", return_config=True).items()
  if "score" in cfg["properties"]
}
```
Or to get all the measures that are metrics (measure of dissimilarity that satisfies the axioms for a distance metric):
```python
metrics = {
  k: similarity.make(f"measure.{k}")
  for k, cfg in similarity.make("measure", return_config=True).items()
  if "metric" in cfg["properties"]
}
```



### Backend Specific Measure
TODO: explain default backend and default parameters


Accessing a measure for a specific backend:
```python
# replace {backend_name} and {metric_name} by the backend and metric names
similarity.make("backend.{backend_name}.metric.{metric_name}")
```


### Common Default Interface
All the measures have a common default interface
Follow the sklean convention.

Two input arraws X, Y of shape sample x neuron.

```python
Measure:
    fit(X, Y)
    score(X, Y) -> float
    fit_score(X: np.ndarray[sample, neuron], Y: np.ndarray[sample, neuron]) -> float

```
Separating `fit` and `score` allows to fit and evaluate the measure on different datasets (e.g. to do cross-validation). `fit_score` provides a quick way to do both on the same data.


### Customized Interface



## Standardized metric interface
TODO: use the term measure insted of metric as it is more general


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


<!-- 
### Why this particular interface?
sklearn -->

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
Here is an overview of the files and directories:
* [similarity/backend](similarity/backend): all the backend folders
* [similarity/measure](similarity/measure): measure cards
* [similarity/processing](similarity/processing): pre- and post-processing functions
* [similarity/api](similarity/api) contains a config file [api.cue](similarity/api/api.cue) that specifies the public api. It also contains a dictionary [api.json](similarity/api/api.json) with all the compiled configs. The `id` argument in `similarity.make` refers to a path in this dictionary, and the corresponding value is used to instantiate the python object returned by the make function.

## Why use CUE instead of plain python?
Can easily generate a json config describing the config

Why cue language? Can use schema to validate config. Show example of adding a metric that doesn't have a card
e.g. it constrains backends can only register metrics that have a card


## Adding an implementation of an existing metric
* create a folder in `similarity/backend`
* create a `requirements.txt` file with the dependencies of the backend. Optionally add a comment with the link to the installation instructions (e.g. in the README of the backend).


Recommend installing CUE extension for vscode. (We recommend this one for now. An official CUE extension is planned to be released soon).

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



<!-- ### Adding a new benchmark
Either copy paste code
* link to commit from which the code was copied
or put the code in a python package and link to it -->


### Adding an new implementation of an existing metric



## References
Leverage implementations from: