import similarity

from time import perf_counter
tic = perf_counter()
similarity.build()
print(f"Built in {perf_counter() - tic:.2f}s")
