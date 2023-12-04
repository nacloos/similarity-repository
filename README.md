Status: Under active development. API subject to change.

<!-- TODO: link survey. Please [share your feedback]()! -->


# Similarity Measures Repository

<!-- ![Backend metrics](figures/backend_metrics.png) -->
![Backend metrics](https://github.com/nacloos/similarity-measures/blob/main/backend_metrics.png)


The goal of this repository is to gather **existing**  implementations of similarity measures for neural networks into a **single** python package, with a **common** and **customizable** interface.


<!-- No measure is re-implemented here! Leverage only existing implementations. -->


## Installation
The python package can be installed with the following command:
```
pip install git+https://github.com/nacloos/similarity-measures.git
```
Or alternatively, you can clone the repository and run `pip install -e .` inside it.


## Getting Started

Here is an example of making and using a measure to quantify the similarity between two datasets.
```python
import numpy as np
import similarity

# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# make a particular measure
measure = similarity.make("measure.procrustes")
score = measure.fit_score(X, Y)
```
`similarity.make` returns a Measure object with the following methods:
* `fit(X, Y)`: fit the measure on the data (useful for cross-validation)
* `score(X, Y) -> float`: evaluate the measure on the data
* `fit_score(X, Y) -> float`: fit and evaluate the measure on the data at the same time

where the inputs `X` and `Y` are 2-dimensional **numpy arrays** of shape **(sample, neuron)**.


Each measure has a unique identifier, where the hyphen `-` symbol is used to specify parameter values for a family of measures. For example, `measure.procrustes-sq-euclidean` refers to the squared Euclidean version of the Procrustes metric.


You can find a list with all the implemented ids [here](similarity/api/__init__.py).


### Selecting Groups of Measures

You can easily make all the available measures by not specifying any particular measure id:
```python	
# make all the measures
measures = similarity.make("measure")
for name, measure in measures.items():
    # all the measures have the same interface
    score = measure.fit_score(X, Y)
    print(f"{name}: {score}")
```

You can also filter out measures based on their properties. For example, to get all the measures that are scoring measures (i.e. measure of similarity where 1 is perfect similarity):

```python
# return_config=True returns the config instead of the instantiated object
measure_configs = similarity.make("measure", return_config=True)
# select desired subset
score_ids = [k for k, cfg in measure_configs.items() if "score" in cfg["properties"]]
# make the measures
score_measures = {k: similarity.make(f"measure.{k}") for k in score_ids}

for name, measure in score_measures.items():
    print(f"Score {name}: {measure.fit_score(X, Y)}")
```

Or to get all the measures that are metrics (i.e. measure of dissimilarity that satisfies the axioms a distance metric):
```python
metric_ids = [k for k, cfg in measure_configs.items() if "metric" in cfg["properties"]]
# make the measures
metric_measures = {k: similarity.make(f"measure.{k}") for k in metric_ids}

for name, measure in metric_measures.items():
    print(f"Metric {name}: {measure.fit_score(X, Y)}")
```


### Choosing a Specific Backend

`similarity.make("measure.{id}")` automatically selects the default backend for the measure. If you want to use a different backend, you can specify it with:
```python
# example of backend and measure
backend_id = "repsim"
measure_id = "procrustes"
measure = similarity.make(f"backend.{backend_id}.measure.{measure_id}")
```

Default backends are specified in [similarity/backend/backends.cue](similarity/backend/backends.cue). In the future, specific criteria such as estimation accuracy or compute efficiency can be used to determine the default backend for a measure.

### Customizing the Measure Interface

The goal of the package is to have a common interface for similarity measures, while keeping this shared interface flexible and customizable. 

For example, if you want a measure that can directly be used as a function instead of a class, you can specify it with:
```python
measure = similarity.make(
  "measure.procrustes",
  interface={
    # replaces the method fit_score with a __call__ method 
    "fit_score": "__call__"
  }
)
score = measure(X, Y)
```

The Measure interface is just a thin wrapper around the backend implementation that converts the inputs to the expected format and allows renaming methods and variables.



### Registering New Measures

You can register new measures locally and use them as any other measure.
For example, to register a function:

```python
def my_metric(x, y):
    return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))

# register the function with a unique id
similarity.register(my_metric, "measure.my_metric.fit_score")

metric = similarity.make("measure.my_metric")
score = metric.fit_score(X, Y)
```

You can also register a class:
```python
class MyMetric:
    def fit(self, X, Y):
        pass

    def score(self, X, Y):
        return X.reshape(-1) @ Y.reshape(-1) / (np.linalg.norm(X) * np.linalg.norm(Y))

    def fit_score(self, x, y):
        self.fit(x, y)
        return self.score(x, y)

similarity.register(MyMetric, "measure.my_metric2")

metric2 = similarity.make("measure.my_metric2")
score = metric2.fit_score(X, Y)
```

<!-- 
## Organization of the Repository
Here is an overview of the files and directories:
* [similarity/backend](similarity/backend): all the backend folders
* [similarity/measure](similarity/measure): measure cards
* [similarity/processing](similarity/processing): pre- and post-processing functions
* [similarity/api](similarity/api) contains a config file [api.cue](similarity/api/api.cue) that specifies the public api. It also contains a dictionary [api.json](similarity/api/api.json) with all the compiled configs. The `id` argument in `similarity.make` refers to a path in this dictionary, and the corresponding value is used to instantiate the python object returned by the make function.

## Why use CUE instead of plain python?
Can easily generate a json config describing the config

Why cue language? Can use schema to validate config. Show example of adding a metric that doesn't have a card
e.g. it constrains backends can only register metrics that have a card


https://cuelang.org/docs/


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



### Adding a new benchmark
Either copy paste code
* link to commit from which the code was copied
or put the code in a python package and link to it


### Adding an new implementation of an existing metric



## References
Leverage implementations from: -->