from collections import defaultdict
import numpy as np
from omegaconf import ListConfig, OmegaConf, DictConfig
from time import perf_counter
import similarity



def papers_with_code():
    papers = similarity.make(package="metric", key="papers")

    total = 0
    with_code = 0
    for name, card in papers.items():
        if "github" in card:
            with_code += 1
        total += 1

    print(f"Papers with code: {with_code}/{total} ({with_code/total*100:.0f}%)")


def generate_data():
    X = np.random.randn(100, 30)
    Y = np.random.randn(100, 30)
    # X = np.random.randn(10, 25)
    # Y = np.random.randn(10, 25)
    return X, Y


# metric = similarity.make("backend.repsim.metric.cka-angular")
# print(metric)
# print(metric._fit_score)
# X, Y = generate_data()
# print(metric.fit_score(X, Y))


def try_metrics():
    metric_names = similarity.make("metric", return_config=True).keys()
    
    for name in metric_names:
        print("Metric:", name)
        # if not name.startswith("rsa-"):
        #     continue

        tic = perf_counter()
        metric = similarity.make(f"metric.{name}")
        print("Time:", perf_counter() - tic)
        print(metric)

        X, Y = generate_data()
        print(metric.fit_score(X=X, Y=Y))

        X = np.random.randn(100, 5, 30)
        Y = np.random.randn(100, 5, 30)
        print(metric.fit_score(X=X, Y=Y))
        print()


    # TODO
    # @config("my_metric")
    # def my_metric(X, Y):
    #     return 0


    # metric = similarity.make(
    #     "my_metric",
    #     preprocessing=["reshape2d"],
    #     postprocessing=["angular_dist_to_score"]
    # )


def backend_consistency_matrix(backend_by_metric, generate_data_fn, n_repeats=10):
    for metric_name, backends in backend_by_metric.items():
        num_backends = len(backends)
        consistency_matrix = np.zeros((num_backends, num_backends))

        X, Y = generate_data_fn()
        # Store backend results in a dictionary
        backend_results = defaultdict(list)
        for k in range(n_repeats):
            Y = X.copy() + np.random.randn(*X.shape) * 0.1 * (k+1) / n_repeats

            for i, backend_name in enumerate(backends):
                metric = similarity.make(f"backend.{backend_name}.metric.{metric_name}")
                result = metric.fit_score(X, Y)
                backend_results[backend_name].append(result)

        backend_results = {
            backend_name: np.array(results)
            for backend_name, results in backend_results.items()
        }

        # Calculate differences and fill the matrix
        for i, backend1 in enumerate(backends):
            for j, backend2 in enumerate(backends):
                if i < j:
                    diff = np.linalg.norm(backend_results[backend1] - backend_results[backend2])
                    # diff = np.max(np.abs(backend_results[backend1] - backend_results[backend2]))
                    consistency_matrix[i, j] = diff
                    consistency_matrix[j, i] = diff  # Symmetric matrix

        print(f"Consistency Matrix for {metric_name}:")
        print(consistency_matrix)
        print("---------------")


def try_backend_consistency():
    backend_by_metric = {
        k: v["backends"]
        for k, v in similarity.make("metric", return_config=True).items()
    }

    # backend_consistency_matrix(backend_by_metric, generate_data)

    X, Y = generate_data()
    for metric_name, backends in backend_by_metric.items():
        metric_results = {}
        print("Metric:", metric_name)
        for backend_name in backends:
            print("Backend:", backend_name)

            metric = similarity.make(f"backend.{backend_name}.metric.{metric_name}")
            assert isinstance(metric, similarity.Metric), f"Expected type Metric, got '{metric}'"
            res = metric.fit_score(X, Y)
            metric_results[backend_name] = res

        print(metric_results)
        print("---------------")


def try_benchmark():
    # TODO: why don't work with config? (argument _out_, problem in instantiate?)
    benchmark = similarity.make("klabunde23_dimensionality")

    for metric_name in ["procrustes", "cca", "svcca", "cka", "rsa"]:
        metric = similarity.make(metric_name)
        print(metric)
        # TODO
        metric_fun = lambda X, Y: metric.fit_score(X, Y)
        benchmark(metric_fun, save_path=f"figures/klabunde23/{metric_name}.png")
        print(benchmark)


if __name__ == "__main__":
    # papers_with_code()
    # try_metrics()
    try_backend_consistency()
    # try_benchmark()
