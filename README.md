# Similarity Measures Repository

Aggregate existing implementations of similarity measures into a single python package.

## Installation

 ```
 pip clone https://github.com/nacloos/similarity-measures.git
 cd similarity-measures
 pip install -e .
 ```

## Usage

```python
import similarity

metric = similarity.make("cca")
```

## Organization of the repository

## Why use CUE instead of plain python?
Can easily generate a json config describing the config

## Contributing
### Adding a new metric
(Or "Registering a new implementation")

Can only add an metric implementation for which there exists a card. Otherwise, create a card first.

Have to only modify the backend folder to add a new backend.
Create a folder for your backend. Create a cue file with the backend package.
Add an import for your backend and add the id/name of the backend in `backends.cue`


### Adding a new benchmark
Either copy paste code
* link to commit from which the code was copied
or put the code in a python package and link to it


### Adding an new implementation of an existing metric

