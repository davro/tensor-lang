from .generators.base import BaseGenerator
from .generators.elementwise import ElementwiseMixin
from .generators.unary import UnaryMixin
from .generators.reductions import ReductionMixin
from .generators.nn import NeuralNetworkMixin
from .generators.shapes import ShapesMixin
from .generators.comparisons import ComparisonsMixin
from .generators.optimizers import OptimizersMixin


class KernelGenerator(
    ElementwiseMixin,
    UnaryMixin,
    ReductionMixin,
    NeuralNetworkMixin,
    ShapesMixin,
    ComparisonsMixin,
    OptimizersMixin,
    BaseGenerator,
):
    def __init__(self, DEBUG_MODE):
        super().__init__(DEBUG_MODE)

    def generate(self, op, *args, **kwargs):
        method_name = f"{op}"
        if not hasattr(self, method_name):
            raise AttributeError(f"KernelGenerator has no generator for op '{op}'")
        return getattr(self, method_name)(*args, **kwargs)

    def supports(self, op):
        return hasattr(self, op)

