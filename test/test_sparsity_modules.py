import torch
import pytest
from torch.nn import MSELoss
from typing import Tuple , Literal
# from custom_linear import CustomLinear  # Import the updated CustomLinear class
import torch
import torch.nn as nn
from typing import Optional, Tuple


class CustomSparseLinear(nn.Linear):
    """
    Custom Linear Layer with Structured Sparsity Support.

    Extends PyTorch's `nn.Linear` to apply controlled sparsity patterns to the weight matrix.
    Supports random sparsity or sparsity focused on non-observable components of the state vector.

    Parameters
    ----------
    `n_hid_vars` : int
        Number of hidden state variables (including observed and hidden variables).

    `n_obs_vars` : int, optional
        Number of observed state variables. especially useful when `mask_type='non_obs_block'`.
        as it is used to define the boundary between observable and non-observable sections.
    
    `bias` : bool, optional
        Whether to include a bias term. Default is False.

    `init_range` : Tuple[float, float], optional
        Range for uniform initialization of weights (and bias if enabled).
        Default is (-0.75, 0.75).

    `mask_type` : Literal['non_obs_block', 'random_block']
        Type of sparsity mask to apply:
        - 'random_block': Applies uniform random sparsity over the entire weight matrix.
        - 'non_obs_block': Applies sparsity only to the non-observable block of the matrix 
          (requires `n_obs_vars` to be set).

    `sparsity` : float, optional
        Proportion of weights to set to zero. Must be in the range [0.0, 1.0].
        Default is 0.0 (no sparsity).



    Direct Attributes (from inputs)
    -------------------------------
    `n_hid_vars` : int
        Number of hidden state variables ( dimensionality of the hidden state variables).

    `n_obs_vars` : int
        Number of observed state variables (dimensionality of the observed state variables).

    `init_range` : Tuple[float, float]
        Range used to initialize weights and optional bias.

    `sparsity` : float
        Desired sparsity level to apply to the weights.

    `mask_type` : str
        Indicates the type of sparsity mask (`random_block` or `non_obs_block`).


    
    Indirect Attributes (initialized internally)
    --------------------------------------------

    `n_state_vars` : int
        Total number of state variables (sum of hidden and observed variables).
        This is used to define the shape of the weight matrix and mask.



    `mask` : torch.Tensor
        Binary mask (shape: `[n_state_vars, n_state_vars]`) indicating which weights are active (1) or zeroed (0),
        based on the chosen `mask_type` and `sparsity`. where n_state_vars =  `n_hid_vars` + `n_obs_vars`

    `weight` : torch.nn.Parameter
        Learnable weight matrix initialized uniformly within `init_range`, then masked.

    `bias` : torch.nn.Parameter or None
        Optional learnable bias vector, also initialized uniformly if present.

    Methods
    -------
    forward(input: torch.Tensor) -> torch.Tensor
        Performs the masked linear transformation on the input tensor.

    Notes
    -----
    - The `non_obs_block` mask applies sparsity only to the part of the matrix
      unrelated to directly observed variables.

    - The mask is enforced during both initialization and backpropagation using hooks.

    - This class is particularly useful where controlled sparsity in the dynamics is desired.
    """

    def __init__(self, n_hid_vars: int, 
                 n_obs_vars: int, 

                 bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75),

                 mask_type: str = Literal['non_obs_block', 'random_block'] ,
                 sparsity: float = 0.0):
        
        self.device =  self._get_default_device()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars  + self.n_obs_vars
        self.init_range = init_range

        super(CustomSparseLinear, self).__init__(in_features= self.n_state_vars, out_features= self.n_state_vars, bias=bias, device=self.device)



        """ 
        Raises
        ------
        ValueError
            If `sparsity` is not between 0 and 1.
        """
        if not (0 <= sparsity <= 1):
            raise ValueError("Sparsity must be between 0 and 1.")
        
        self.sparsity = sparsity
        self.mask_type = mask_type


        # Initialize weights and biases
        self._initialize_weights()

        # Generate the mask based on the mask type
        self.mask = self._generate_mask()

        # Apply the mask to the weights
        self._apply_mask()

        # Register hook to enforce the mask during backpropagation
        self.weight.register_hook(self._enforce_mask)

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

    def _initialize_weights(self):
        """
        Initializes weights and biases within the specified range.
        """
        nn.init.uniform_(self.weight.data, self.init_range[0], self.init_range[1])
        if self.bias is not None:
            nn.init.uniform_(self.bias.data, self.init_range[0], self.init_range[1])


    def _generate_mask(self) -> torch.Tensor:
        """
        Generates the sparsity mask based on the specified `mask_type`.

        Returns
        -------
        torch.Tensor
            Binary mask of shape `(n_state_vars, n_state_vars)`.

        Raises
        ------
        ValueError
            If an invalid `mask_type` is provided.
        """
        mask = torch.ones(self.n_state_vars, self.n_state_vars, device=self.device)

        if self.mask_type == "random_block":  
            # Random sparsity over the entire weight matrix
            total_weights = self.n_state_vars * self.n_state_vars

            zeroed_weights = int(self.sparsity * total_weights)

            random_indices = torch.randperm(total_weights, device=self.device)[:zeroed_weights]

            flat_mask = mask.view(-1)
            flat_mask[random_indices] = 0.0

            mask = flat_mask.view(self.n_state_vars, self.n_state_vars)

        elif self.mask_type == "non_obs_block":

            if self.n_obs_vars is None or not (0 < self.n_obs_vars < self.n_state_vars):
                raise ValueError("`n_obs_vars` must be provided and satisfy 0 < n_obs_vars < non_obs_block for `non_obs_block`.")

            # Split the weight matrix into two blocks
            obs_block = torch.ones((self.n_state_vars, self.n_obs_vars), device=self.device)  # (n_hid_vars, p)
            non_obs_block = torch.ones((self.n_state_vars, self.n_state_vars - self.n_obs_vars), device=self.device)  # (n_hid_vars, n_hid_vars - p)

            # Apply sparsity only on the non-observable block
            total_weights_non_obs = non_obs_block.numel()

            zeroed_weights = int(self.sparsity * total_weights_non_obs)

            random_indices = torch.randperm(total_weights_non_obs, device=self.device)[:zeroed_weights]

            flat_non_obs = non_obs_block.view(-1)

            flat_non_obs[random_indices] = 0.0

            non_obs_block = flat_non_obs.view(self.n_state_vars, self.n_state_vars - self.n_obs_vars)

            # Combine the blocks to form the mask
            mask = torch.cat((obs_block, non_obs_block), dim=1)

        else:
            raise ValueError(f"Invalid mask_type: {self.mask_type}. Must be 'random_block' or 'non_obs_block'.")

        return mask

    def _apply_mask(self):
        """
        Applies the mask to the weight matrix.
        """
        self.weight.data *= self.mask

    def _enforce_mask(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Hook to enforce the mask during backpropagation.

        Ensures that:
        - The zeroed-out weights remain zeroed and are not updated.
        - The gradients corresponding to zeroed-out weights are also set to zero.

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the loss with respect to the weight matrix.

        Returns
        -------
        torch.Tensor
            Modified gradient respecting the mask.
        """
        with torch.no_grad():
            # Enforce the mask on the weights
            self.weight.data *= self.mask
        # Zero out gradients for masked weights
        return grad * self.mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomLinear layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_state_vars)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_state_vars)`.
        """
        return nn.functional.linear(input, self.weight, self.bias)

import torch
import pytest
from torch.nn import MSELoss

def test_initialization_random_block():
    """Test initialization with random_block mask."""
    n_obs, n_hid = 4, 6
    sparsity = 0.3
    layer = CustomSparseLinear(n_hid, n_obs, mask_type="random_block", sparsity=sparsity)

    assert layer.weight.shape == (n_obs + n_hid, n_obs + n_hid)
    assert torch.all((layer.mask == 0) | (layer.mask == 1)), "Mask should be binary."
    assert torch.sum(layer.mask == 0) >= int(sparsity * layer.n_state_vars ** 2), "Sparsity not applied properly."

def test_initialization_non_obs_block():
    """Test initialization with non_obs_block mask."""
    n_obs, n_hid = 5, 5
    sparsity = 0.5
    layer = CustomSparseLinear(n_hid, n_obs, mask_type="non_obs_block", sparsity=sparsity)

    total = layer.n_state_vars
    obs_block = layer.mask[:, :n_obs]
    non_obs_block = layer.mask[:, n_obs:]

    assert torch.all(obs_block == 1), "Observables block should be fully connected."
    assert torch.sum(non_obs_block == 0) >= int(sparsity * non_obs_block.numel()), "Sparsity not properly enforced on non-observable block."

def test_invalid_sparsity():
    """Test that invalid sparsity raises error."""
    with pytest.raises(ValueError, match="Sparsity must be between 0 and 1"):
        CustomSparseLinear(n_hid_vars=4, n_obs_vars=2, sparsity=1.5)

def test_invalid_mask_type():
    """Test that invalid mask type raises error."""
    with pytest.raises(ValueError, match="Invalid mask_type:"):
        CustomSparseLinear(n_hid_vars=4, n_obs_vars=2, mask_type="invalid_type")

def test_forward_output_shape():
    """Test output shape of the forward pass."""
    n_obs, n_hid = 5, 7
    layer = CustomSparseLinear(n_hid_vars = n_hid, n_obs_vars =  n_obs , mask_type="random_block", sparsity=0.2)
    x = torch.randn(10, n_obs + n_hid)  # (batch_size, n_state_vars)
    y = layer(x)
    assert y.shape == x.shape, "Output shape mismatch"

def test_weight_masking_applied():
    """Ensure masked weights are zero after init."""
    n_obs, n_hid = 6, 4
    layer = CustomSparseLinear(n_hid, n_obs, mask_type="random_block", sparsity=0.8)
    masked_weights = layer.weight.data * (1 - layer.mask)
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights)), "Masked weights should be zero."

def test_enforce_mask_on_backward():
    """Test that gradients respect the sparsity mask during backward."""
    n_obs, n_hid = 4, 4
    layer = CustomSparseLinear(n_hid, n_obs, mask_type="random_block", sparsity=0.5)
    x = torch.randn(3, layer.n_state_vars, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    grad_masked = layer.weight.grad * (1 - layer.mask)
    assert torch.allclose(grad_masked, torch.zeros_like(grad_masked)), "Masked gradient values should be zero."

def test_gradient_flow():
    """Test gradient flow through the module."""
    n_obs, n_hid = 4, 8
    layer = CustomSparseLinear(n_hid, n_obs ,mask_type="non_obs_block" , sparsity=0.0)
    x = torch.randn(2, layer.n_state_vars, requires_grad=True)
    output = layer(x)
    target = torch.ones_like(output)
    loss = MSELoss()(output, target)
    loss.backward()

    assert x.grad is not None, "Gradients should flow back to input"
    assert layer.weight.grad is not None, "Gradients should flow back to weights"

def test_edge_case_zero_sparsity():
    """Ensure full connection when sparsity is 0."""
    n_obs, n_hid = 3, 3
    layer = CustomSparseLinear(n_hid, n_obs,  mask_type="random_block", sparsity=0.0)
    assert torch.all(layer.mask == 1), "Mask should be all ones for zero sparsity"

def test_edge_case_full_sparsity_random_block():
    """Ensure all weights are zeroed for full sparsity (random_block)."""
    n_obs, n_hid = 3, 3
    layer = CustomSparseLinear(n_hid, n_obs, sparsity=1.0, mask_type="random_block")
    assert torch.all(layer.weight == 0), "All weights should be zero with full sparsity"

def test_forward_pass_consistency():
    """Ensure forward pass is deterministic given fixed mask and input."""
    n_obs, n_hid = 3, 3
    layer = CustomSparseLinear(n_hid, n_obs, mask_type="random_block", sparsity=0.25)
    x = torch.randn(1, n_obs + n_hid)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2), "Forward pass should be deterministic"
