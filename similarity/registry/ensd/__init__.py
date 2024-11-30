# https://github.com/camillerb/ENSD
# https://www.biorxiv.org/content/10.1101/2023.07.27.550815v1.full.pdf
from .ENSD_Tutorial import ENSD, computeDist

import similarity


similarity.register("ensd/ensd", ENSD, preprocessing=["center_columns", "transpose"])
similarity.register("ensd/computeDist", computeDist, preprocessing=["center_columns", "transpose"])
