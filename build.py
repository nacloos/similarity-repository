import similarity

from time import perf_counter
tic = perf_counter()
similarity.build()
print("Time:", perf_counter() - tic)
