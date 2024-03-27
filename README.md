# Similarity Repository


![Backend metrics](https://github.com/nacloos/similarity-repository/blob/main/figures/implemented_measures.png)


The goal of this repository is to gather **existing**  implementations of similarity measures for neural networks into a **single** python package, with a **common** and **customizable** interface. This repository is not meant to provide new implementations. It only refers to existing ones.


## Installation
The python package can be installed with the following command:
```
pip install git+https://github.com/nacloos/similarity-repository.git
```
Or alternatively, you can clone the repository and run `pip install -e .` inside it.


## Getting Started

Each similarity measure has a unique identifier that can be used to create a measure object.

```python
import numpy as np
import similarity

# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# make a measure object
measure = similarity.make("measure.netrep.procrustes-angular")
score = measure(X, Y)
```

In this example, the full id of the measure is `"measure.netrep.procrustes-angular"`. The id is composed of three parts: 
* the object type (`measure`)
* the backend id (`netrep`)
* the measure id (`procrustes-angular`)

You can find a list with all the implemented ids [here](similarity/types/__init__.py).


### Selecting Groups of Measures

You can easily select all the measures implemented for a specific backend by using the wildcard `*`:
```python
measures = similarity.make("measure.netrep.*")
for name, measure in measures.items():
    score = measure(X, Y)
    print(f"{name}: {score}")
```

You can also select all the backend implementations for a specific measure:
```python
measures = similarity.make("measure.*.procrustes-angular")
for name, measure in measures.items():
    score = measure(X, Y)
    print(f"{name}: {score}")
```

### Registering New Measures

You can register a new measure locally:
```python
# register the function with a unique id
@similarity.register("measure.my_package.my_measure", function=True)
def my_measure(x, y):
    return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))
```

And then use it as any other measure:
```python
measure = similarity.make("measure.my_package.my_measure")
score = measure(X, Y)
```

You can also register a class:
```python
@similarity.register("measure.my_package.my_measure2")
class MyMeasure:
    def fit(self, X, Y):
        self.X_norm = np.linalg.norm(X)
        self.Y_norm = np.linalg.norm(Y)

    def score(self, X, Y):
        return X.reshape(-1) @ Y.reshape(-1) / (self.X_norm * self.Y_norm)

    def fit_score(self, x, y):
        self.fit(x, y)
        return self.score(x, y)

    def __call__(self, x, y):
        return self.fit_score(x, y)
```

The advantage of using a class is that you can separate the fitting and scoring steps:
```python
measure2 = similarity.make("measure.my_package.my_measure2")

X_fit, Y_fit = np.random.randn(100, 30), np.random.randn(100, 30)
X_val, Y_val = np.random.randn(100, 30), np.random.randn(100, 30)

measure2.fit(X_fit, Y_fit)
score = measure2.score(X_val, Y_val)
```


<!-- ## Contributing
See backend folder for examples of how to register new measures. -->


 ## References
Haruo Yanai. 1974. Unification of Various Techniques of Multivariate Analysis by Means of Generalized Coefficient of Determination. Kodo Keiryogaku (The Japanese Journal of Behaviormetrics) 1 (1974).

Frances Ding, Jean-Stanislas Denain, and Jacob Steinhardt. 2021. Grounding Representation Similarity Through Statistical Testing. In NeurIPS.

Max Klabunde, Tobias Schumacher, Markus Strohmaier, & Florian Lemmerich. (2023). Similarity of Neural Network Models: A Survey of Functional and Representational Measures.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. 2019. Similarity of Neural Network Representations Revisited. In ICML.

Nikolaus Kriegeskorte, Marieke Mur, and Peter Bandettini. 2008. Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience 2 (2008).

Richard D. Lange, Devin Kwok, Jordan Matelsky, Xinyue Wang, David S. Rolnick, & Konrad P. Kording. (2022). Neural Networks as Paths through the Space of Representations.

Richard D. Lange, David S. Rolnick, and Konrad P. Kording. 2022. Clustering units in neural networks: upstream vs downstream information. TMLR (2022).

Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John E. Hopcroft. 2016. Convergent Learning: Do different neural networks learn the same representations?. In ICLR.

Ari S. Morcos, Maithra Raghu, and Samy Bengio. 2018. Insights on representational similarity in neural networks with canonical correlation. In NeurIPS.

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. In NeurIPS.

Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J. Majaj, Rishi Rajalingham, Elias B. Issa, Kohitĳ Kar, Pouya Bashivan, Jonathan Prescott-Roy, Franziska Geiger, Kailyn Schmidt, Daniel L. K. Yamins, & James J. DiCarlo (2018). Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?. bioRxiv preprint.

Schrimpf, M., Kubilius, J., Lee, M., Murty, N., Ajemian, R., & DiCarlo, J. (2020). Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence. Neuron.

Mahdiyar Shahbazi, Ali Shirali, Hamid Aghajan, and Hamed Nili. 2021. Using distance on the Riemannian manifold to compare representations in brain and in models. NeuroImage 239 (2021).

Anton Tsitsulin, Marina Munkhoeva, Davide Mottin, Panagiotis Karras, Alex Bronstein, Ivan Oseledets, and Emmanuel Mueller. 2020. The Shape of Data: Intrinsic Distance for Data Distributions. In ICLR.

Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John E. Hopcroft. 2018. Towards Understanding Learning Representations: To What Extent Do Different Neural Networks Learn the Same Representation. In NeurIPS.

Alex H. Williams, Erin Kunz, Simon Kornblith, and Scott W. Linderman. 2021. Generalized Shape Metrics on Neural Representations. In NeurIPS.

