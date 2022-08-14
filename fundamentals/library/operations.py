from numpy import ndarray

"""
BLUEPRINT for sending inputs forward and gradients backward, with the shapes of what they recieve on the forward pass matching the shapes of what they recieve on the backward pass
"""


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
    an operation with parameters
    """
    def __init__(self, param: ndarray) -> ndarray:
        super().__init_()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:

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
Layers
"""

