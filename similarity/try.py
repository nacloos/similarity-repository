import numpy as np
import similarity

# TODO: select backend
# metric = similarity.make("procrustes", backend="netrep")


# TODO: one Metric object with score and distance methods?
# metric.fit_score()
# metric.fit_distance()
# netrep: call method score even though it outputs a distance...


# TODO: how to get all the backends in one config?
# test = similarity.make("metric/glob")
# print(test)

def try_metrics():
    names = ["procrustes", "cca", "svcca", "cka", "rsa"]

    for name in names:
        print("Metric:", name)
        metric = similarity.make(f"metric/{name}")
        print(metric)

        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        print(metric.fit_score(X=X, Y=Y))

        X = np.random.randn(100, 5, 10)
        Y = np.random.randn(100, 5, 10)
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
    try_metrics()
    # try_benchmark()
