"""
Custom linear layers for HCNN models.

This module provides specialized linear layers with custom initialization
and device management for HCNN architectures.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


class CustomLinear(nn.Linear):
    """
    Custom Linear layer with user-defined weight initialization.

    This class extends PyTorch's `nn.Linear` module to allow users to specify a custom
    range for uniform weight initialization. It is primarily used in HCNN
    for mapping between variables (`state-to-state`, `external-to-state`).

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool, default=False
        Whether to include a bias term
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Range for uniform weight initialization
    device : Optional[torch.device], default=None
        Device to place the layer on. If None, uses automatic detection

    Attributes
    ----------
    init_range : Tuple[float, float]
        Stores the initialization range for weights and bias
    device : torch.device
        Device where the layer parameters are stored

    Notes
    -----
    This class ensures consistent initialization when used in architectures that are 
    sensitive to the starting parameter values for the weights matrix and bias, such as HCNNs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        device: Optional[torch.device] = None
    ):
        # Store init_range first
        self.init_range = init_range

        # Standard PyTorch device semantics: build on the default device (CPU)
        # unless an explicit device is passed, then let the caller place the whole
        # model with ``.to(device)``. (Previously this layer auto-detected and
        # grabbed MPS/CUDA at construction, which silently split a model across
        # devices because the base model kept s0 on CPU.)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device
        )

        # Initialize weights and biases with custom range
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and bias with uniform distribution in init_range."""
        with torch.no_grad():
            nn.init.uniform_(self.weight, self.init_range[0], self.init_range[1])
            if self.bias is not None:
                nn.init.uniform_(self.bias, self.init_range[0], self.init_range[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomLinear layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, in_features)`

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, out_features)`
        """
        return nn.functional.linear(input, self.weight, self.bias)

    def reset_parameters(self):
        """Reset parameters using custom initialization."""
        self._initialize_parameters()

    def extra_repr(self) -> str:
        """Return extra representation string for the layer."""
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, init_range={self.init_range}'
        )


class DiagonalMatrix(nn.Linear):
    """
    Custom Linear Layer with Learnable Diagonal Weight Matrix for HCNN models.

    This module enforces a learnable diagonal structure on the weight matrix,
    making it suitable for HCNN formulations. The layer maintains diagonal structure
    throughout training with robust numerical stability for large matrices.

    Features:
    - Diagonal elements are trainable and clamped to [0, 1] range
    - Off-diagonal elements are always zero and receive no gradient updates
    - Robust gradient handling to prevent NaN/exploding values
    - Two initialization modes: all_one and all_random

    Parameters
    ----------
    n_features : int
        Number of features (both input and output, as this is a square matrix)
    bias : bool, default=False
        Whether to include a bias term in the linear transformation
    init_mode : Literal['all_one', 'all_random'], default='all_one'
        Initialization mode:
        - 'all_one': Initialize all diagonal elements to 1.0
        - 'all_random': Initialize diagonal elements randomly in [0, 1]
        Ignored if init_diag is specified
    init_diag : Optional[float], default=None
        If specified, initialize all diagonal elements to this value
        Takes precedence over init_mode for backward compatibility
    device : Optional[torch.device], default=None
        Device to place the layer on

    Attributes
    ----------
    n_features : int
        Number of features
    init_mode : str
        Initialization mode used
    diagonal_mask : torch.Tensor
        Binary mask to enforce diagonal structure (1 on diagonal, 0 elsewhere)
    """

    def __init__(
        self,
        n_features: int,
        bias: bool = False,
        init_mode: Literal['all_one', 'all_random'] = 'all_one',
        init_diag: Optional[float] = None,
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = 0.5
    ):
        # Validate inputs
        if init_mode not in ['all_one', 'all_random']:
            raise ValueError(f"init_mode must be 'all_one' or 'all_random', got {init_mode}")

        # Set attributes before calling super().__init__() to avoid issues with reset_parameters.
        # Device follows standard PyTorch semantics (build on default device unless
        # given); no auto-grab of MPS/CUDA.
        self.n_features = n_features
        self.init_mode = init_mode
        self.init_diag = init_diag
        self.grad_clip = grad_clip  # max grad-norm for the diagonal; None disables clipping

        super().__init__(
            in_features=n_features,
            out_features=n_features,
            bias=bias,
            device=device
        )

        # Create diagonal mask (1 on diagonal, 0 elsewhere) on the weight's device
        self.register_buffer(
            'diagonal_mask',
            torch.eye(n_features, device=self.weight.device, dtype=self.weight.dtype),
            persistent=False
        )

        # Initialize parameters
        self._initialize_parameters()

        # Register backward hook to enforce diagonal structure and prevent NaNs
        self._register_gradient_hook()

    def _initialize_parameters(self):
        """Initialize diagonal matrix with specified initialization mode or init_diag value."""
        with torch.no_grad():
            dev, dt = self.weight.device, self.weight.dtype
            # Zero out the entire weight matrix first
            self.weight.data.zero_()

            # If init_diag is specified, use it directly (for backward compatibility)
            if self.init_diag is not None:
                diagonal_values = torch.full((self.n_features,), self.init_diag,
                                           device=dev, dtype=dt)
            elif self.init_mode == 'all_one':
                # Set diagonal elements to 1.0
                diagonal_values = torch.ones(self.n_features, device=dev, dtype=dt)
            elif self.init_mode == 'all_random':
                # Set diagonal elements to random values in [0, 1]
                diagonal_values = torch.rand(self.n_features, device=dev, dtype=dt)
            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

            # Set diagonal elements
            self.weight.data.fill_diagonal_(0.0)  # Ensure clean start
            for i in range(self.n_features):
                self.weight.data[i, i] = diagonal_values[i]

            # Initialize bias if present
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def _register_gradient_hook(self):
        """Register backward hook to enforce diagonal structure and prevent NaN values."""
        def gradient_hook(grad):
            if grad is not None:
                # Zero out off-diagonal gradients (keep the matrix diagonal)
                masked_grad = grad * self.diagonal_mask.to(dtype=grad.dtype, device=grad.device)

                # Prevent NaN and inf values
                masked_grad = torch.where(
                    torch.isfinite(masked_grad),
                    masked_grad,
                    torch.zeros_like(masked_grad)
                )

                # Optional gradient-norm clipping for the diagonal (configurable via
                # grad_clip; None disables). Default 0.5 is conservative - relax it
                # when studying the memory-gate dynamics.
                if self.grad_clip is not None:
                    grad_norm = torch.norm(masked_grad)
                    if grad_norm > self.grad_clip:
                        masked_grad = masked_grad * (self.grad_clip / (grad_norm + 1e-8))

                # Note: Weight clamping is done in forward pass to avoid interfering with gradients
                return masked_grad
            return grad

        # Register the hook
        self.weight.register_hook(gradient_hook)

    def forward(self, input: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Apply diagonal linear transformation to input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, n_features)
        verbose : bool, default=False
            If True, prints the current diagonal values

        Returns
        -------
        torch.Tensor
            Transformed output tensor
        """
        # Enforce diagonal structure before forward pass
        with torch.no_grad():
            self.weight.data *= self.diagonal_mask

            # Ensure values stay in [0, 1] range with epsilon bounds
            epsilon = 1e-6
            self.weight.data.clamp_(min=epsilon, max=1.0 - epsilon)

        if verbose:
            print(f"Diagonal weights: {self.get_diagonal_values()}")

        return nn.functional.linear(input, self.weight, self.bias)

    def get_diagonal_values(self) -> torch.Tensor:
        """Get current diagonal values."""
        return torch.diag(self.weight)

    def set_diagonal_values(self, values: torch.Tensor):
        """
        Set diagonal values directly.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape (n_features,) with values to set on diagonal
        """
        if values.shape != (self.n_features,):
            raise ValueError(f"Expected values shape ({self.n_features},), got {values.shape}")

        with torch.no_grad():
            # Zero out matrix and set diagonal
            self.weight.data.zero_()
            for i in range(self.n_features):
                self.weight.data[i, i] = torch.clamp(values[i], min=1e-6, max=1.0 - 1e-6)

    def enforce_diagonal_structure(self):
        """
        Explicitly enforce diagonal structure and value bounds.

        This method should be called after optimizer steps to ensure the matrix
        maintains its diagonal structure and value constraints.
        """
        with torch.no_grad():
            # Apply diagonal mask
            self.weight.data *= self.diagonal_mask

            # Clamp values to [0, 1] with epsilon bounds for numerical stability
            epsilon = 1e-6
            self.weight.data.clamp_(min=epsilon, max=1.0 - epsilon)

    def get_structure_info(self) -> dict:
        """
        Get information about the current diagonal structure.

        Returns
        -------
        dict
            Dictionary containing structure statistics
        """
        total_params = self.weight.numel()
        diagonal_params = self.n_features
        off_diagonal_params = total_params - diagonal_params

        # Check how many off-diagonal elements are actually zero
        mask_tensor = torch.as_tensor(self.diagonal_mask, dtype=torch.float32)
        off_diagonal_mask = (1 - mask_tensor).bool()  # True for off-diagonal positions
        off_diagonal_weights = self.weight[off_diagonal_mask]
        zero_off_diagonal = (torch.abs(off_diagonal_weights) < 1e-8).sum().item()

        diagonal_values = self.get_diagonal_values()

        return {
            'total_parameters': total_params,
            'diagonal_parameters': diagonal_params,
            'off_diagonal_parameters': off_diagonal_params,
            'zero_off_diagonal': zero_off_diagonal,
            'structure_maintained': zero_off_diagonal == off_diagonal_params,
            'diagonal_min': diagonal_values.min().item(),
            'diagonal_max': diagonal_values.max().item(),
            'diagonal_mean': diagonal_values.mean().item(),
            'init_mode': self.init_mode,
            'values_in_bounds': ((diagonal_values >= 0) & (diagonal_values <= 1)).all().item()
        }

    def reset_parameters(self):
        """Reset parameters using the original initialization mode."""
        # Only initialize if diagonal_mask exists (i.e., after full initialization)
        if hasattr(self, 'diagonal_mask'):
            self._initialize_parameters()
        else:
            # Default initialization for parent class during super().__init__()
            super().reset_parameters()

    def extra_repr(self) -> str:
        """Return extra representation string for the layer."""
        return (
            f'n_features={self.n_features}, bias={self.bias is not None}, '
            f'init_mode={self.init_mode}'
        )
