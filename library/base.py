import numpy as np
from numpy import ndarray

from typing import List


def assert_same_shape(array: ndarray, array_grad: ndarray):
    assert (
        array.shape == array_grad.shape
    ), """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(
        tuple(array_grad.shape), tuple(array.shape)
    )
    return None


class Operation(object):
    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        """
        Stores input_ in the input variable
        Calls the self._output function
        """

        self.input = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        calls the self._input_grad functions
        checks that the appropriate shapes match
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input, self.input_grad)
        return self.input_grad

    # custom forward computation for each function
    def _output(self) -> ndarray:
        """
        the _output method must be defined for each operation
        """
        raise NotImplementedError

    # custom backward computation for each function
    def _input_grad(self, output_grad) -> ndarray:
        """
        the _input_grad method must be defined for each operation
        """
        raise NotImplementedError


class ParamOperation(Operation):
    """
    An Operation with parameters
    """

    def __init__(self, param: ndarray) -> ndarray:
        """
        ParamOperation method
        """
        super().__init_()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input_grad and self._param_grad. Also checks for appropriate shapes.
        """

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        every subclass of ParamOperation must implement _param_grad
        """
        raise NotImplementedError


"""
Specific Operations
"""


class WeightMultiply(ParamOperation):
    """
    Weight multiplication operation for a neural network.
    """

    def __init__(self, W: ndarray):
        """
        Initialize Operation with self.param = W.
        """
        super().__init__(W)

    def _output(self) -> ndarray:
        """
        Compute output.
        """
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        """
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient.
        """
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    """
    Compute bias addition.
    """

    def __init__(self, B: ndarray):
        """
        Initialize Operation with self.param = B.
        Check appropriate shape.
        """
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> ndarray:
        """
        Compute output.
        """
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        """
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute parameter gradient.
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):
    """
    Sigmoid activation function.
    """

    def __init__(self) -> None:
        """
        Pass
        """
        super().__init__()

    def _output(self) -> ndarray:
        """
        Compute output.
        """
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient.
        """
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Linear(Operation):
    """
    Identity activation function
    """

    def __init__(self) -> None:
        """
        Pass
        """
        super().__init__()

    def _output(self) -> ndarray:
        """
        Pass through
        """
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Pass through
        """
        return output_grad
