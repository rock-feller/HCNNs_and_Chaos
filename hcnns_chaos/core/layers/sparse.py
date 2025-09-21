"""
Sparse linear layers for large-scale HCNN models.

This module provides linear layers with structured sparsity patterns
for efficient computation in high-dimensional dynamical systems.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


class CustomSparseLinear(nn.Linear):
    """
    Custom Linear Layer with Structured Sparsity Support.

    Extends PyTorch's `nn.Linear` to apply controlled sparsity patterns to the weight matrix.
    Supports random sparsity or sparsity focused on non-observable components of the state vector.
    Maintains sparsity throughout training by masking gradients of zeroed weights.

    Parameters
    ----------
    n_obs_vars : int
        Number of observed state variables
    n_hid_vars : int
        Number of hidden state variables
    bias : bool, default=False
        Whether to include a bias term
    init_range : Tuple[float, float], default=(-0.75, 0.75)
        Range for uniform initialization of weights
    mask_type : Literal['non_obs_block', 'random_block'], default='random_block'
        Type of sparsity mask to apply:
        - 'random_block': Applies uniform random sparsity over the entire weight matrix
        - 'non_obs_block': Applies sparsity only to the non-observable block of the matrix
    sparsity_ratio : float, default=0.0
        Proportion of weights to set to zero. Must be in the range [0.0, 1.0)
    device : Optional[torch.device], default=None
        Device to place the layer on

    Attributes
    ----------
    n_obs_vars : int
        Number of observed variables
    n_hid_vars : int
        Number of hidden variables
    n_state_vars : int
        Total number of state variables (n_obs_vars + n_hid_vars)
    init_range : Tuple[float, float]
        Weight initialization range
    mask_type : str
        Type of sparsity mask applied
    sparsity_ratio : float
        Sparsity ratio applied
    sparsity_mask : torch.Tensor
        Binary mask defining sparsity pattern (1 = keep weight, 0 = zero weight)

    Raises
    ------
    ValueError
        If sparsity_ratio is not in [0, 1)
        If mask_type is not supported
    """

    def __init__(
        self,
        n_obs_vars: int,
        n_hid_vars: int,
        bias: bool = False,
        init_range: Tuple[float, float] = (-0.75, 0.75),
        mask_type: Literal['non_obs_block', 'random_block'] = 'random_block',
        sparsity_ratio: float = 0.0,
        device: Optional[torch.device] = None
    ):
        # Validate inputs
        if not (0 <= sparsity_ratio < 1):
            raise ValueError("sparsity_ratio must be in the range [0, 1).")

        if mask_type not in ['non_obs_block', 'random_block']:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        # Get device if not provided
        if device is None:
            device = self._get_default_device()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = n_hid_vars + n_obs_vars
        self.init_range = init_range
        self.mask_type = mask_type
        self.sparsity_ratio = sparsity_ratio
        self.device = device

        super().__init__(
            in_features=self.n_state_vars,
            out_features=self.n_state_vars,
            bias=bias,
            device=device
        )

        # Create and apply sparsity mask
        self._create_sparsity_mask()
        # Re-initialize parameters with sparsity
        self._initialize_parameters()

        # Register backward hook to maintain sparsity during training
        self._register_gradient_hook()

    def _get_default_device(self) -> torch.device:
        """Get the default device for computation."""
        from ...utils.device import get_device
        return get_device()

    def _create_sparsity_mask(self):
        """Create sparsity mask based on mask_type and sparsity ratio."""
        if self.sparsity_ratio == 0.0:
            # No sparsity - all ones mask
            mask = torch.ones(
                self.n_state_vars, self.n_state_vars,
                device=self.device, dtype=torch.float32
            )
        elif self.mask_type == 'random_block':
            # Random sparsity across entire matrix
            mask = self._create_random_mask()
        elif self.mask_type == 'non_obs_block':
            # Sparsity only in non-observable block
            mask = self._create_non_obs_block_mask()
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        # Register as buffer (non-trainable parameter)
        self.register_buffer('sparsity_mask', mask, persistent=False)

    def _create_random_mask(self) -> torch.Tensor:
        """Create random sparsity mask across entire weight matrix."""
        total_elements = self.n_state_vars * self.n_state_vars
        num_zeros = int(total_elements * self.sparsity_ratio)

        # Create mask with all ones
        mask = torch.ones(total_elements, device=self.device, dtype=torch.float32)

        # Randomly select indices to zero out
        if num_zeros > 0:
            zero_indices = torch.randperm(total_elements, device=self.device)[:num_zeros]
            mask[zero_indices] = 0.0

        # Reshape to matrix form
        return mask.view(self.n_state_vars, self.n_state_vars)

    def _create_non_obs_block_mask(self) -> torch.Tensor:
        """
        Create sparsity mask only in non-observable blocks.

        The weight matrix is structured as:
        [obs_to_obs,  obs_to_hid ]
        [hid_to_obs,  hid_to_hid ]

        Sparsity is applied to obs_to_hid and hid_to_hid blocks.
        obs_to_obs and hid_to_obs blocks remain dense.
        """
        # Start with all ones
        mask = torch.ones(
            self.n_state_vars, self.n_state_vars,
            device=self.device, dtype=torch.float32
        )

        if self.n_hid_vars > 0 and self.sparsity_ratio > 0.0:
            # Calculate elements to sparsify (obs_to_hid + hid_to_hid blocks)
            obs_to_hid_elements = self.n_obs_vars * self.n_hid_vars
            hid_to_hid_elements = self.n_hid_vars * self.n_hid_vars
            sparsifiable_elements = obs_to_hid_elements + hid_to_hid_elements

            num_zeros = int(sparsifiable_elements * self.sparsity_ratio)

            if num_zeros > 0:
                # Create indices for obs_to_hid and hid_to_hid blocks
                all_indices = []

                # obs_to_hid block: rows [0:n_obs_vars], cols [n_obs_vars:n_state_vars]
                for i in range(self.n_obs_vars):
                    for j in range(self.n_obs_vars, self.n_state_vars):
                        all_indices.append(i * self.n_state_vars + j)

                # hid_to_hid block: rows [n_obs_vars:n_state_vars], cols [n_obs_vars:n_state_vars]
                for i in range(self.n_obs_vars, self.n_state_vars):
                    for j in range(self.n_obs_vars, self.n_state_vars):
                        all_indices.append(i * self.n_state_vars + j)

                # Randomly select indices to zero out
                if len(all_indices) >= num_zeros:
                    zero_positions = torch.randperm(len(all_indices), device=self.device)[:num_zeros]
                    for pos in zero_positions:
                        linear_idx = all_indices[pos]
                        i, j = divmod(linear_idx, self.n_state_vars)
                        mask[i, j] = 0.0

        return mask

    def _register_gradient_hook(self):
        """Register backward hook to mask gradients and prevent NaN values."""
        def gradient_hook(grad):
            if grad is not None:
                # Convert mask to proper tensor type
                mask_tensor = torch.as_tensor(self.sparsity_mask, dtype=grad.dtype, device=grad.device)

                # Mask gradients for sparse weights (zero out gradients for sparse positions)
                masked_grad = grad * mask_tensor

                # Prevent NaN and inf values
                masked_grad = torch.where(
                    torch.isfinite(masked_grad),
                    masked_grad,
                    torch.zeros_like(masked_grad)
                )

                # Adaptive gradient clipping to prevent exploding gradients
                grad_norm = torch.norm(masked_grad)
                max_grad_norm = 1.0  # Conservative clipping threshold

                if grad_norm > max_grad_norm:
                    masked_grad = masked_grad * (max_grad_norm / (grad_norm + 1e-8))

                return masked_grad
            return grad

        # Register the hook
        self.weight.register_hook(gradient_hook)

    def reset_parameters(self) -> None:
        """Override to prevent automatic initialization during super().__init__()."""
        # Only initialize if sparsity_mask exists (i.e., after _create_sparsity_mask)
        if hasattr(self, 'sparsity_mask'):
            self._initialize_parameters()
        else:
            # Default initialization for parent class
            super().reset_parameters()

    def _initialize_parameters(self):
        """Initialize weights with custom range and apply sparsity mask."""
        with torch.no_grad():
            # Use Xavier/Glorot initialization scaled to init_range for better stability
            fan_in = self.n_state_vars
            fan_out = self.n_state_vars
            std = (2.0 / (fan_in + fan_out)) ** 0.5

            # Scale to desired range
            range_scale = (self.init_range[1] - self.init_range[0]) / 2.0
            range_center = (self.init_range[1] + self.init_range[0]) / 2.0

            # Initialize with scaled normal distribution for better stability
            nn.init.normal_(self.weight, mean=range_center, std=std * range_scale)

            # Clamp to ensure values stay within init_range
            self.weight.data.clamp_(self.init_range[0], self.init_range[1])

            # Apply sparsity mask
            mask_tensor = torch.as_tensor(self.sparsity_mask, dtype=torch.float32)
            self.weight.data *= mask_tensor

            # Initialize bias if present
            if self.bias is not None:
                nn.init.uniform_(self.bias, self.init_range[0], self.init_range[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparsity enforcement.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_state_vars)`

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_state_vars)`
        """
        # Apply sparsity mask to weights before forward pass
        # This ensures sparsity is maintained even if gradients somehow update sparse weights
        with torch.no_grad():
            mask_tensor = torch.as_tensor(self.sparsity_mask, dtype=torch.float32)
            self.weight.data *= mask_tensor

        return nn.functional.linear(input, self.weight, self.bias)

    def get_sparsity_info(self) -> dict:
        """
        Get information about the current sparsity pattern.

        Returns
        -------
        dict
            Dictionary containing sparsity statistics
        """
        total_params = self.weight.numel()
        zero_params = (self.weight == 0).sum().item()
        actual_sparsity = zero_params / total_params

        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'non_zero_parameters': total_params - zero_params,
            'target_sparsity_ratio': self.sparsity_ratio,
            'actual_sparsity': actual_sparsity,
            'mask_type': self.mask_type,
            'matrix_shape': (self.n_state_vars, self.n_state_vars),
            'obs_vars': self.n_obs_vars,
            'hid_vars': self.n_hid_vars
        }

    def visualize_sparsity_pattern(self) -> torch.Tensor:
        """
        Get the current sparsity pattern for visualization.

        Returns
        -------
        torch.Tensor
            Binary tensor showing sparsity pattern (1 = non-zero, 0 = zero)
        """
        return (self.weight != 0).float()

    def enforce_sparsity(self):
        """
        Explicitly enforce sparsity pattern on weights.

        This method can be called during training to ensure sparsity is maintained,
        especially useful after optimizer steps.
        """
        with torch.no_grad():
            mask_tensor = torch.as_tensor(self.sparsity_mask, dtype=torch.float32)
            self.weight.data *= mask_tensor

    def get_effective_sparsity(self) -> float:
        """
        Calculate the effective sparsity ratio of the current weights.

        Returns
        -------
        float
            Actual sparsity ratio (proportion of zero weights)
        """
        total_params = self.weight.numel()
        zero_params = (torch.abs(self.weight) < 1e-8).sum().item()
        return zero_params / total_params

    def extra_repr(self) -> str:
        """Return extra representation string for the layer."""
        return (
            f'n_state_vars={self.n_state_vars}, '
            f'n_obs_vars={self.n_obs_vars}, '
            f'n_hid_vars={self.n_hid_vars}, '
            f'bias={self.bias is not None}, '
            f'sparsity_ratio={self.sparsity_ratio}, '
            f'mask_type={self.mask_type}'
        )
