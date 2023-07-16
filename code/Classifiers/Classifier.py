import numpy
import abc

class Classifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, DTR: numpy.ndarray, LTR: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pass

    @abc.abstractmethod
    def compute_scores(self, DTE: numpy.ndarray, evaluation_function: callable, **kwargs):
        pass

