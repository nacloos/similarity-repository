# important to import registration first so that make and register can be imported from similarity
from .registration import (
    make,
    register,
    all_measures,
    all_papers,
    match,
    wrap_measure,
)

from .types import IdType

from . import processing
from . import registry
from . import papers

from .standardization import register_standardized_measures
register_standardized_measures()
