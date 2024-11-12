# important to import registration first so that make and register can be imported from similarity
from .registration import make, register, is_registered, match, MeasureInterface, Measure

from .types import MeasureIdType, BackendIdType

from . import processing
from . import backend
from . import cards
from . import papers

# error if import transforms here, do it in registration.make instead
# from . import transforms
