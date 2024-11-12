"""
Register similarity measures from different backends.
"""
from . import yuanli2333
from . import kornblith19
from . import netrep
from . import mklabunde
from . import rsatoolbox
from . import repsim
from . import nn_similarity_index
from . import sim_metric
from . import deepdive
# from . import brainscore
from . import svcca
from . import subspacematch
from . import rtd
from . import pyrcca
from . import imd
from . import dsa
from . import platonic

# packages that already use similarity-repository to register their measures
# import diffscore
