# important to import registration first so that make and register can be imported from similarity
from .registration import (
    make,
    register,
    all_measures,
    is_registered,
    match,
    MeasureInterface,
    Measure
)

from .types import MeasureIdType, BackendIdType

from . import backend
# from . import registry
from . import processing
# from . import cards
from . import papers

# error if import transforms here, do it in registration.make instead
# from . import transforms
