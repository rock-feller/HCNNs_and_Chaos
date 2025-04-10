import pytest
import torch
from torch.nn import MSELoss
from typing import Tuple, Optional, Literal
import torch.nn as nn




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
          (requires `n_obs_var` to be set).

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
        based on the chosen `mask_type` and `sparsity`. where n_state_vars =  `n_hid_vars` + `n_obs_var`

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
                raise ValueError("`n_obs_var` must be provided and satisfy 0 < n_obs_vars < non_obs_block for `non_obs_block`.")

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









class LargeSparse_cell(nn.Module):

    """
    Large Sparse HCNN Cell.

    Implements a scalable variant of the Historical Consistent Neural Network (HCNN) cell
    designed for high-dimensional dynamical systems using sparsity-aware non linear transformation.
    It performs a state-to-state mapping using sparsity-aware non linear transformation
    and produces outputs based on hidden states.

    This cell supports structured sparsity for efficient computation and also supports
    teacher forcing during training.

    Parameters
        ----------
    `n_hid_vars` : int
        Number of hidden state variables (including observed and hidden variables).

    `n_obs_var` : int, optional
        Number of observed state variables. especially useful when `mask_type='non_obs_block'`.
        as it is used to define the boundary between observable and non-observable sections.
    
    `init_range` : Tuple[float, float]
        Range used to initialize weights and optional bias.

    `sparsity_ratio` : float
        Desired sparsity level to apply to the weights.

    `mask_type` : str
        Indicates the type of sparsity mask (`random_block` or `non_obs_block`).


    `device` : torch.device
        Device on which operations are performed and tensors are stored.

    Indirect Attributes (initialized internally)
    --------------------------------------------

    `n_state_vars` : int
        Total number of state variables (sum of hidden and observed variables).
        This is used to define the shape of the weight matrix and mask.

    `Sparse_A`: `CustomSparseLinear`
        A Custom Linear Layer with Structured Sparsity Support.

    `ConMat` : torch.Tensor
        A readout matrix that maps hidden states to observed outputs.
        Initialized as horizontal concatenation of an identity matrix of shape (`n_obs_var`, `n_obs_var`).
        and a zero matrix of  shape (`n_obs_var`, `n_state_vars`). 
        Registered as a non-persistent buffer.

    `Ide` : torch.Tensor
            Identity matrix used for internal computations on the hidden state.
            Shape is (`n_state_vars`, `n_state_vars`). Registered as a non-persistent buffer.
    
    `mask` : torch.Tensor
        Binary mask (shape: `[n_state_vars, n_state_vars]`) indicating which weights are active (1) or zeroed (0),
        based on the chosen `mask_type` and `sparsity`. where n_state_vars =  `n_hid_vars` + `n_obs_var`

    `weight` : torch.nn.Parameter
        Learnable weight matrix initialized uniformly within `init_range`, then masked.

    `bias` : torch.nn.Parameter or None
        Optional learnable bias vector, also initialized uniformly if present.

        

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the Large Sparse HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """


    def __init__(self, n_obs_vars: int, n_hid_vars: int, bias: bool = False, init_range: Tuple[float, float] = (-0.75, 0.75),
                 mask_type: str = "random_block", sparsity_ratio: float = 0.0 ):
    


        super(LargeSparse_cell, self).__init__()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars
        self.device = self._get_default_device()
        self.bias = bias
        # Initialize sparse transformation
        self.Sparse_A = CustomSparseLinear(n_hid_vars = self.n_hid_vars, n_obs_vars=self.n_obs_vars,
                                            bias=self.bias, init_range=init_range,
                                           sparsity=sparsity_ratio, mask_type=mask_type
                                           )
        
        self.register_buffer(name = 'ConMat', 
                            tensor= torch.eye(self.n_obs_vars, self.n_state_vars), 
                            persistent = False)
        

        self.register_buffer(name = 'Ide', 
                            tensor= torch.eye(self.n_state_vars ),
                            persistent = False)

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def forward(self, state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        r'''
        Forward pass of the Vanilla HCNN Cell.

        Parameters
        ----------
        `state` : torch.Tensor | shape = (`n_state_vars`,)

            The current state tensor  `$\mathbf{s}_{t}$` is the HCNN state vector of shape (`n_state_vars`,)

        `teacher_forcing` : bool | default=False

            Whether to use teacher forcing for the state transition:

            - `True`: Use the provided observation (`observation`) to guide the state transition.
            - `False`: Compute the next state based on the model's prediction.

        `observation` : Optional[torch.Tensor], default=None | shape=(`n_obs_vars`,)

            The ground-truth observation tensor (`y_true`) corresponds to the observable at time t. Required when 
            `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

            - expectation : torch.Tensor

                Predicted observation tensor (`y_hat`) |  shape (`n_obs_vars`,).

            - next_state : torch.Tensor

                Updated state tensor (`$\mathbf{s}_{t+1}$`) |  shape (`n_state_vars`,).

            - delta_term : `y_hat - y_true` | shape (n_obs_var,) | Optional , can be None.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the method uses `observation` to compute a correction term 
        (`delta_term`) for guiding the state transition.
        
        - If `teacher_forcing` is False, the state transition is based purely on the model's 
        internal computation.
        '''
        # Compute expected output (y_hat)
        expectation = torch.matmul(state , self.ConMat.T)

        if teacher_forcing:

            if observation is None:

                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            
            delta_term = observation - expectation
            
    
            teach_forc = torch.matmul(delta_term,self.ConMat)
            
            r_state = state - teach_forc
            
            next_state = self.Sparse_A(torch.tanh(r_state))
            
            return expectation, next_state, delta_term
       
        else:
    
            r_state = torch.matmul( state,self.Ide)
            
            next_state = self.Sparse_A(torch.tanh(r_state))
            
            return expectation, next_state, None



import torch
import pytest
from torch.nn import MSELoss


def test_largesparse_initialization():
    cell = LargeSparse_cell(n_obs_vars=4, n_hid_vars=6, sparsity_ratio=0.3)
    assert hasattr(cell, "Sparse_A"), "Missing Sparse_A attribute"
    assert cell.ConMat.shape == (4, 10), "Incorrect shape for ConMat"
    assert cell.Ide.shape == (10, 10), "Incorrect shape for Ide"


def test_forward_autonomous_mode():
    cell = LargeSparse_cell(n_obs_vars=3, n_hid_vars=5)
    state = torch.randn(8)
    expectation, next_state, delta_term = cell(state, teacher_forcing=False)

    assert expectation.shape == (3,), "Incorrect expectation shape"
    assert next_state.shape == (8,), "Incorrect next state shape"
    assert delta_term is None, "Delta term should be None when teacher_forcing is False"


def test_forward_teacher_forcing_mode():
    cell = LargeSparse_cell(n_obs_vars=3, n_hid_vars=5)
    state = torch.randn(8)
    observation = torch.randn(3)

    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    assert expectation.shape == (3,), "Incorrect expectation shape"
    assert next_state.shape == (8,), "Incorrect next state shape"
    assert delta_term.shape == (3,), "Incorrect delta term shape"
    assert torch.allclose(delta_term, observation - expectation), "Delta term mismatch"


def test_teacher_forcing_requires_observation():
    cell = LargeSparse_cell(n_obs_vars=3, n_hid_vars=5)
    state = torch.randn(8)
    with pytest.raises(ValueError):
        _ = cell(state, teacher_forcing=True)


def test_nogradient_flow_teacher_forcing():
    cell = LargeSparse_cell(n_obs_vars=3, n_hid_vars=5)
    state = torch.randn(8, requires_grad=True)
    observation = torch.randn(3)
    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)
    loss = MSELoss()(expectation, observation)
    loss.backward()

    assert state.grad is not None, "No gradient for input state"
    assert cell.Sparse_A.weight.grad is  None, "No gradient should flow for Sparse_A weights"

def test_gradient_flow_teacher_forcing():
    cell = LargeSparse_cell(n_obs_vars=3, n_hid_vars=5)
    state = torch.randn(8, requires_grad=True)
    observation = torch.randn(3)
    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)
    loss = MSELoss()(next_state, torch.randn(8))
    loss.backward()

    assert state.grad is not None, "No gradient for input state"
    assert cell.Sparse_A.weight.grad is  not None, "gradient should flow for Sparse_A weights"



def test_sparsity_random_mask_applied():
    cell = LargeSparse_cell(n_obs_vars=2, n_hid_vars=4, sparsity_ratio=1.0, mask_type="random_block")
    num_nonzero = torch.count_nonzero(cell.Sparse_A.weight).item()
    assert num_nonzero == 0, "Expected full sparsity"




def test_sparsity_non_obs_only_shape():
    cell = LargeSparse_cell(n_obs_vars=2, n_hid_vars=4, sparsity_ratio=0.25, mask_type="non_obs_block")
    assert cell.Sparse_A.mask.shape == (6, 6), "Incorrect mask shape for non_obs_only"


def test_forward_consistency_on_same_input():
    cell = LargeSparse_cell(n_obs_vars=2, n_hid_vars=4)
    state = torch.randn(6)
    obs = torch.randn(2)

    out1 = cell(state, teacher_forcing=True, observation=obs)
    out2 = cell(state, teacher_forcing=True, observation=obs)

    for a, b in zip(out1, out2):
        if a is not None and b is not None:
            assert torch.allclose(a, b), "Inconsistent outputs across repeated calls"


def test_autonomous_step_predictive_validity():
    cell = LargeSparse_cell(n_obs_vars=2, n_hid_vars=4)
    state = torch.randn(6, requires_grad=True)
    y_pred, next_state, _ = cell(state, teacher_forcing=False)
    loss = MSELoss()(y_pred, torch.randn(2))
    loss.backward()

    assert state.grad is not None, "Gradients not backpropagated in autonomous mode"


# def test_initialization():
#     """Test if the LargeSparse_cell initializes correctly."""
#     n_obs = 4
#     n_hid_vars = 6
#     sparsity = 0.5
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity)

#     # Check attributes
#     assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
#     assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
#     assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
#     assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
#     assert isinstance(cell.Sparse_A, CustomSparseLinear), "Sparse_A is not properly initialized."


# def test_device_assignment():
#     """Test if the default device is assigned correctly."""
#     cell = LargeSparse_cell(4, 6)
#     expected_device = cell._get_default_device()
#     assert cell.device == expected_device, "Device assignment is incorrect."


# def test_sparsity_random_mask():
#     """Test if random sparsity mask is applied correctly in the CustomSparseLinear layer."""
#     n_obs = 4
#     n_hid_vars = 6
#     sparsity = 0.3
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="random")

#     mask = cell.Sparse_A.mask
#     assert mask is not None, "Mask should be created for random sparsity."
#     num_weights = n_hid_vars * n_hid_vars
#     num_zeros = torch.sum(mask == 0).item()
#     expected_zeros = int(num_weights * sparsity)
#     assert num_zeros == expected_zeros, "Incorrect number of zeroed weights in the mask."


# def test_sparsity_non_obs_only():
#     """Test if non_obs_only mask is applied correctly."""
#     n_obs = 3
#     n_hid_vars = 5
#     sparsity = 0.4
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="non_obs_only", p=n_obs)

#     mask = cell.Sparse_A.mask
#     assert mask is not None, "Mask should be created for non_obs_only."
#     assert mask[:, :n_obs].sum() == n_hid_vars * n_obs, "Observable part of the mask should be dense."
#     hidden_mask = mask[:, n_obs:]
#     num_zeros = torch.sum(hidden_mask == 0).item()
#     num_weights = hidden_mask.numel()
#     expected_zeros = int(num_weights * sparsity)
#     assert num_zeros == expected_zeros, "Incorrect number of zeroed weights in the hidden part of the mask."


# def test_forward_no_teacher_forcing():
#     """Test the forward pass without teacher forcing."""
#     n_obs, n_hid_vars = 4, 6
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     expectation, next_state, delta_term = cell(state, teacher_forcing=False)

#     # Check output shapes
#     assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
#     assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
#     assert delta_term is None, "Delta term should be None when teacher forcing is False."


# def test_forward_with_teacher_forcing():
#     """Test the forward pass with teacher forcing."""
#     n_obs, n_hid_vars = 4, 6
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.2)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     observation = torch.randn(n_obs)  # Random observation
#     expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

#     # Check output shapes
#     assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
#     assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
#     assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."


# def test_teacher_forcing_without_observation():
#     """Test if an error is raised when teacher forcing is enabled but observation is missing."""
#     n_obs, n_hid_vars = 4, 6
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

#     state = torch.randn(n_hid_vars)  # Random initial state

#     with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
#         _ = cell(state, teacher_forcing=True)


# def test_gradient_flow():
#     """Test if gradients flow properly through the model."""
#     n_obs, n_hid_vars = 4, 6
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.1)

#     state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
#     some_target_state = torch.randn(n_hid_vars)  # Random target state
#     observation = torch.randn(n_obs)  # Random observation

#     expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

#     loss_fct = MSELoss()
#     loss = loss_fct(next_state, some_target_state)
#     loss.backward()  # Backpropagation

#     assert state.grad is not None, "Gradients are not flowing through the initial state."
#     assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
#     assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
#         "Gradients for masked weights should be zero."


# def test_sparsity_retention_during_training():
#     """Test that the sparsity mask is retained during weight updates."""
#     n_obs, n_hid_vars = 4, 6
#     sparsity = 0.2
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity)

#     optimizer = torch.optim.SGD(cell.parameters(), lr=0.01)

#     for _ in range(5):
#         state = torch.randn(n_hid_vars, requires_grad=True)
#         observation = torch.randn(n_obs)

#         expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)
#         loss_fct = MSELoss()
#         loss = loss_fct(expectation, observation)
#         loss.backward()
#         optimizer.step()

#         # Check that the sparsity mask is retained
#         with torch.no_grad():
#             assert torch.allclose(cell.Sparse_A.weight * cell.Sparse_A.mask, cell.Sparse_A.weight), \
#                 "Sparsity mask is not retained during training."

# def test_recurrent_behavior_largesparse():
#     """Test the Large Sparse HCNN Cell in a recurrent setup over a sequence."""
#     T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

#     # Initialize input sequence (T, n_obs)
#     tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
#     state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
#     loss_fct = MSELoss()

#     # Use lists to store outputs
#     list_yhats = []
#     list_states = [state]

#     for time_step in range(T - 1):
#         y_hat, next_state, _ = cell(
#             state=list_states[time_step],
#             teacher_forcing=True,
#             observation=tens_ytrues[time_step]
#         )
#         list_yhats.append(y_hat)
#         list_states.append(next_state)

#     # Compute y_hat for the last time step
#     last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
#     list_yhats.append(last_y_hat)

#     # Stack and compute loss
#     tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
#     loss = loss_fct(tens_yhats, tens_ytrues)
#     loss.backward()

#     # Assertions
#     assert state.grad is not None, "Gradients are not flowing through the initial state."
#     assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
#     assert loss.item() > 0, "Loss should be a positive scalar."
#     assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
#         "Gradients for masked weights should be zero."


# def test_recurrent_behavior_largesparse_across_epochs():
#     """Test the Large Sparse HCNN Cell in a recurrent setup over multiple epochs."""
#     T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.2, mask_type="non_obs_only",p=n_obs)

#     # Initialize input sequence (T, n_obs)
#     tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
#     state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
#     loss_fct = MSELoss()
#     optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)

#     for epoch in range(5):  # Test over 5 epochs
#         list_yhats = []
#         list_states = [state]
#         optimizer.zero_grad()

#         for time_step in range(T - 1):
#             y_hat, next_state, _ = cell(
#                 state=list_states[time_step],
#                 teacher_forcing=True,
#                 observation=tens_ytrues[time_step]
#             )
#             list_yhats.append(y_hat)
#             list_states.append(next_state)

#         # Compute y_hat for the last time step
#         last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
#         list_yhats.append(last_y_hat)

#         # Stack and compute loss
#         tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
#         loss = loss_fct(tens_yhats, tens_ytrues)
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch + 1}:")
#         print(f"Sparse_A weight gradients:\n{cell.Sparse_A.weight.grad}")
#         print(f"Sparse_A weights after update:\n{cell.Sparse_A.weight}")
#         print("=" * 50)

#         # Assertions
#         assert state.grad is not None, "Gradients are not flowing through the initial state."
#         assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
#         assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
#             "Gradients for masked weights should be zero."
#         assert loss.item() > 0, "Loss should be a positive scalar."


# def test_mask_retention_recurrent_training():
#     """Test that the sparsity mask is retained during recurrent training."""
#     T, n_obs, n_hid_vars = 3, 4, 6
#     sparsity = 0.3
#     cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="non_obs_only", p=n_obs)

#     optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)
#     tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations

#     for epoch in range(3):  # Test for 3 epochs
#         state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
#         list_yhats = []
#         list_states = [state]
#         optimizer.zero_grad()

#         for time_step in range(T - 1):
#             y_hat, next_state, _ = cell(
#                 state=list_states[time_step],
#                 teacher_forcing=True,
#                 observation=tens_ytrues[time_step]
#             )
#             list_yhats.append(y_hat)
#             list_states.append(next_state)

#         # Compute y_hat for the last time step
#         last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
#         list_yhats.append(last_y_hat)

#         # Compute loss and update
#         tens_yhats = torch.stack(list_yhats)
#         loss = MSELoss()(tens_yhats, tens_ytrues)
#         loss.backward()
#         optimizer.step()

#         # Ensure the sparsity mask is retained
#         with torch.no_grad():
#             assert torch.allclose(cell.Sparse_A.weight * cell.Sparse_A.mask, cell.Sparse_A.weight), \
#                 "Sparsity mask is not retained during recurrent training."


# # Run the test suite
# if __name__ == "__main__":
#     pytest.main([__file__])

