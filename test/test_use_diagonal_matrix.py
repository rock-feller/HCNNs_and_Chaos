import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import MSELoss
import unittest




class DiagonalMatrix(nn.Linear):
    """
    Custom Linear Layer with Learnable Diagonal Weight Matrix.

    This module enforces a learnable diagonal structure on the weight matrix,
    making it suitable for the LSTM formulation of HCNN. 
    The layer clamps learned diagonal values to the [0, 1] range
    and ensures that all off-diagonal elements remain zero throughout training.

    Parameters
    ----------
    `n_state_vars` : int
        Total number of state variables (sum of hidden and observed variables).
        This is used to define the shape of the weight matrix and mask.

    `bias` : bool, optional
        Whether to include a bias term in the linear transformation. Default is False.

    `init_diag` : float, optional
        Initial value for the diagonal elements of the weight matrix. If set to `None`,
        random values close to zero are used instead. Default is 1.0.

    Notes
    ------
    
        Only `n_state_vars` is provided. It will hold the same value as `in_vars` and `out_vars`, as a diagonal matrix must be square.

    Direct Attributes (from inputs)
    -------------------------------
    `n_state_vars` : int
        Input dimensionality of the layer (same as output).

    `out_vars` : int
        Output dimensionality of the layer (same as input).

    `bias` : bool
        Whether a bias term is included in the transformation.

    `init_diag` : Optional[float]
        Value used to initialize the diagonal of the weight matrix.

    Indirect Attributes (initialized internally)
    --------------------------------------------

    `n_state_vars`: int
        Total number of state variables (sum of hidden and observed variables).
        This is used to define the shape of the weight matrix and mask.

    `weight` : torch.Tensor
        Weight matrix of shape `(n_state_vars, n_state_vars)` with non-zero values only along the diagonal.
        Clamped to range [0, 1].

    `mask` : torch.Tensor
        A binary matrix used to zero out off-diagonal values and clamp gradients during backpropagation.

    Methods
    -------
    forward(input: torch.Tensor, verbose: bool = False) -> torch.Tensor
        Applies the diagonal linear transformation to the input tensor. Optionally prints the diagonal
        values if `verbose=True`.

    Notes
    -----
    - The diagonal values are clamped in-place to the [0, 1] range after every backward pass.
    - A hook is registered to ensure off-diagonal elements remain zero throughout training.
    - This module is ideal for scenarios requiring independent scaling of each variable.
    """

    def __init__(self, n_state_vars: int, bias: bool = False,
                  init_diag: Optional[float] = 1.0):

        
        super(DiagonalMatrix, self).__init__(in_features =n_state_vars, out_features = n_state_vars, bias=bias)

        self.n_state_vars = n_state_vars
        self.device  = self._get_default_device()
        # Initialize the weight matrix
        nn.init.constant_(self.weight, 0)  # Set all weights to zero

        if init_diag is not None:
            self.weight.data.fill_diagonal_(init_diag)  # User-specified or default value

        else:
            rand_diag = torch.empty(self.n_state_vars ).uniform_(-1e-5, 1e-5)
            self.weight.data.fill_diagonal_(0.0)
            self.weight.data += torch.diag(rand_diag)
        # Pre-compute and store the diagonal mask
        
        self.register_buffer("mask", torch.eye(self.n_state_vars , device=self.weight.device))

        # Register hook
        self.weight.register_hook(self._clamp_and_zero_out)

    def _get_default_device(self) -> torch.device:
        """
        Determines the default device to use for computations.

        Returns
        -------
        torch.device
            The default device (`cuda`, `mps`, or `cpu`).
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            # 
    def _clamp_and_zero_out(self, grad):

        """
        Gradient hook that:
        - Zeros out off-diagonal elements in the weight matrix
        - Clamps diagonal values to [0, 1]
        - Applies gradient mask to ensure only diagonal updates

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the loss with respect to the weight matrix.

        Returns
        -------
        torch.Tensor
            Masked gradient tensor.
        """

        with torch.no_grad():
            self.weight.data = torch.mul(self.weight.data ,self.mask)  # Retain only diagonal elements
            self.weight.data.clamp_(min=0.0, max=1.0)

        return grad * self.mask  # Mask gradients to zero out off-diagonal elements
    

    def forward(self, input: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Applies the diagonal linear transformation to the input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        verbose : bool, optional
            If True, prints the current diagonal values. Default is False.

        Returns
        -------
        torch.Tensor
            Transformed output tensor.
        """
        if verbose:
            print(f"Diagonal weights: {torch.diag(self.weight)}")

        return nn.functional.linear(input, self.weight, self.bias)



def test_initialization_with_fixed_diag():
    """Ensure initialization with fixed diagonal sets correct values."""
    dim = 5
    init_val = 0.6
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=init_val)
    weight = layer.weight.data
    assert torch.allclose(torch.diag(weight), torch.full((dim,), init_val)), \
        "Diagonal values are not initialized correctly."
    assert torch.all(weight == torch.diag(torch.diag(weight))), \
        "Off-diagonal elements should be zero."


def test_initialization_with_random_diag():
    """Ensure random diagonal initialization is close to zero."""
    dim = 5
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=None)
    weight = layer.weight.data
    diag_vals = torch.diag(weight)
    assert torch.all(diag_vals.abs() < 1e-4), \
        "Randomly initialized diagonal values should be near zero."
    assert torch.all(weight == torch.diag(diag_vals)), \
        "Off-diagonal elements should be zero."


def test_forward_pass_identity_behavior():
    """Forward pass with init_diag=1.0 should behave like identity."""
    dim = 4
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=1.0)
    x = torch.randn(2, dim)
    output = layer(x)
    assert torch.allclose(x, output), "Forward pass should act as identity."


def test_clamp_and_zero_behavior_during_backprop():
    """Check clamping and masking after backward pass."""
    dim = 6
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=0.9)
    input_tensor = torch.randn(2, dim, requires_grad=True)
    target = torch.randn(2, dim)

    output = layer(input_tensor)
    loss = MSELoss()(output, target)
    loss.backward()

    # Check gradient masking
    grad_masked = layer.weight.grad
    assert torch.all(grad_masked == torch.mul(grad_masked, layer.mask)), \
        "Gradient should be masked to diagonal only."

    # Check weight clamping
    diag_vals = torch.diag(layer.weight)
    assert torch.all((diag_vals >= 0.0) & (diag_vals <= 1.0)), \
        "Diagonal weights must be clamped within [0, 1]."

    # Check off-diagonal weights are still zero
    assert torch.all(layer.weight.data == torch.diag(torch.diag(layer.weight))), \
        "Off-diagonal weights should remain zero after backprop."


def test_device_assignment():
    """Ensure correct device assignment."""
    layer = DiagonalMatrix(n_state_vars=3)
    expected = torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")
    assert layer.device == expected, "Incorrect default device assignment."


def test_gradient_flow_to_weights():
    """Ensure gradients properly flow through weights."""
    dim = 4
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=0.8)
    input_tensor = torch.randn(3, dim, requires_grad=True)
    target = torch.randn(3, dim)

    output = layer(input_tensor)
    loss = MSELoss()(output, target)
    loss.backward()

    assert input_tensor.grad is not None, "Gradient did not flow to input."
    assert layer.weight.grad is not None, "Gradient did not flow to weights."


def test_verbose_output(capsys):
    """Ensure verbose prints the correct diagonal values."""
    dim = 3
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=0.9)
    x = torch.ones((1, dim))
    _ = layer(x, verbose=True)
    captured = capsys.readouterr()
    assert "Diagonal weights" in captured.out, "Verbose output did not print."


def test_batch_input_consistency():
    """Ensure the layer handles batch inputs correctly."""
    dim = 5
    layer = DiagonalMatrix(n_state_vars=dim, init_diag=0.5)
    x = torch.ones((10, dim))
    out = layer(x)
    expected = 0.5 * x
    assert torch.allclose(out, expected), "Batch input not handled correctly."
