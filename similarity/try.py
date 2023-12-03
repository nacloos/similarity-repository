from collections import defaultdict
import numpy as np
from omegaconf import ListConfig, OmegaConf, DictConfig
from time import perf_counter
import similarity



def papers_with_code():
    papers = similarity.make(package="measure", key="papers")

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


# measure = similarity.make("backend.repsim.measure.cka-angular")
# print(measure)
# print(measure._fit_score)
# X, Y = generate_data()
# print(measure.fit_score(X, Y))


def try_measures():
    measure_names = similarity.make("measure", return_config=True).keys()
    
    for name in measure_names:
        print("measure:", name)
        # if not name.startswith("rsa-"):
        #     continue

        tic = perf_counter()
        measure = similarity.make(f"measure.{name}")
        print("Time:", perf_counter() - tic)
        print(measure)

        X, Y = generate_data()
        print(measure.fit_score(X=X, Y=Y))

        X = np.random.randn(100, 5, 30)
        Y = np.random.randn(100, 5, 30)
        print(measure.fit_score(X=X, Y=Y))
        print()


    # TODO
    # @config("my_measure")
    # def my_measure(X, Y):
    #     return 0


    # measure = similarity.make(
    #     "my_measure",
    #     preprocessing=["reshape2d"],
    #     postprocessing=["angular_dist_to_score"]
    # )


def backend_consistency_matrix(backend_by_measure, generate_data_fn, n_repeats=10):
    for measure_name, backends in backend_by_measure.items():
        num_backends = len(backends)
        consistency_matrix = np.zeros((num_backends, num_backends))

        X, Y = generate_data_fn()
        # Store backend results in a dictionary
        backend_results = defaultdict(list)
        for k in range(n_repeats):
            Y = X.copy() + np.random.randn(*X.shape) * 0.1 * (k+1) / n_repeats

            for i, backend_name in enumerate(backends):
                measure = similarity.make(f"backend.{backend_name}.measure.{measure_name}")
                result = measure.fit_score(X, Y)
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
                    consistency_matrix[j, i] = diff  # Symmeasure matrix

        print(f"Consistency Matrix for {measure_name}:")
        print(consistency_matrix)
        print("---------------")


def try_backend_consistency():
    backend_by_measure = {
        k: v["backends"]
        for k, v in similarity.make("measure", return_config=True).items()
    }
    # backend_consistency_matrix(backend_by_measure, generate_data)

    X, Y = generate_data()
    for measure_name, backends in backend_by_measure.items():
        measure_results = {}
        print("measure:", measure_name)
        for backend_name in backends:
            print("Backend:", backend_name)

            measure = similarity.make(f"backend.{backend_name}.measure.{measure_name}")
            assert isinstance(measure, similarity.Measure), f"Expected type measure, got '{measure}'"
            res = measure.fit_score(X, Y)
            measure_results[backend_name] = res

        print(measure_results)
        print("---------------")


def try_benchmark():
    # TODO: why don't work with config? (argument _out_, problem in instantiate?)
    benchmark = similarity.make("klabunde23_dimensionality")

    for measure_name in ["procrustes", "cca", "svcca", "cka", "rsa"]:
        measure = similarity.make(measure_name)
        print(measure)
        # TODO
        measure_fun = lambda X, Y: measure.fit_score(X, Y)
        benchmark(measure_fun, save_path=f"figures/klabunde23/{measure_name}.png")
        print(benchmark)


if __name__ == "__main__":
    # papers_with_code()
    # try_measures()
    try_backend_consistency()
    # try_benchmark()
