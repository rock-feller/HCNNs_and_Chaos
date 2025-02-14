import torch
from typing import Optional , Tuple
import torch.nn as nn
# import pytest
from torch.nn import MSELoss

class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        """CustomLinear layer with user-defined weight initialization range."""
        self.init_range = init_range
        super(CustomLinear, self).__init__(in_features, out_features, bias=bias)

        # Properly initialize weights and biases
        nn.init.uniform_(self.weight.data, self.init_range[0], self.init_range[1])
        if bias:
            nn.init.uniform_(self.bias.data, self.init_range[0], self.init_range[1])



class CustomSparseLinear(nn.Linear):
    def __init__(self, n_hid_vars: int, 
                 bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 sparsity: float = 0.0,
                 mask_type: str = "random",
                 p: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        CustomLinear layer with sparsity applied based on the mask type.

        Parameters
        ----------
        n_hid_vars : int
            Number of hidden variables (both input and output features).
        bias : bool, optional
            If True, includes a bias term. Default is False.
        init_range : Tuple[float, float], optional
            Range for uniform initialization of weights. Default is (-0.75, 0.75).
        sparsity : float, optional
            Proportion of weights to set to zero (random sparsity). Default is 0.0 (no sparsity).
        mask_type : str, optional
            Type of mask to apply. Options:
            - "random": Random sparsity over the entire weight matrix.
            - "non_obs_only": Sparsity applied only on the non-observable block of the weight matrix.
        p : int, optional
            Number of observable variables when using `non_obs_only` mask type.
        device : torch.device, optional
            The device to use for computations. If not specified, automatically selects
            `cuda`, `mps`, or `cpu` based on availability.

        Notes
        -----
        - For `random`, the entire weight matrix has a proportion of its elements randomly zeroed out.
        - For `non_obs_only`, sparsity is applied to the lower-right block of the weight matrix
          (size `(n_hid_vars - p, n_hid_vars)`) with `p` observable variables.
        """
        self.device = device if device else self._get_default_device()

        super(CustomSparseLinear, self).__init__(n_hid_vars, n_hid_vars, bias=bias, device=self.device)

        self.init_range = init_range

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
        self.p = p
        self.n_hid_vars = n_hid_vars

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
            Binary mask of shape `(n_hid_vars, n_hid_vars)`.

        Raises
        ------
        ValueError
            If an invalid `mask_type` is provided.
        """
        mask = torch.ones(self.n_hid_vars, self.n_hid_vars, device=self.device)

        if self.mask_type == "random":
            # Random sparsity over the entire weight matrix
            total_weights = self.n_hid_vars * self.n_hid_vars
            zeroed_weights = int(self.sparsity * total_weights)
            random_indices = torch.randperm(total_weights, device=self.device)[:zeroed_weights]
            flat_mask = mask.view(-1)
            flat_mask[random_indices] = 0.0
            mask = flat_mask.view(self.n_hid_vars, self.n_hid_vars)

        elif self.mask_type == "non_obs_only":
            if self.p is None or not (0 < self.p < self.n_hid_vars):
                raise ValueError("`p` must be provided and satisfy 0 < p < n_hid_vars for `non_obs_only`.")

            # Split the weight matrix into two blocks
            obs_block = torch.ones((self.n_hid_vars, self.p), device=self.device)  # (n_hid_vars, p)
            non_obs_block = torch.ones((self.n_hid_vars, self.n_hid_vars - self.p), device=self.device)  # (n_hid_vars, n_hid_vars - p)

            # Apply sparsity only on the non-observable block
            total_weights_non_obs = non_obs_block.numel()
            zeroed_weights = int(self.sparsity * total_weights_non_obs)
            random_indices = torch.randperm(total_weights_non_obs, device=self.device)[:zeroed_weights]
            flat_non_obs = non_obs_block.view(-1)
            flat_non_obs[random_indices] = 0.0
            non_obs_block = flat_non_obs.view(self.n_hid_vars, self.n_hid_vars - self.p)

            # Combine the blocks to form the mask
            mask = torch.cat((obs_block, non_obs_block), dim=1)

        else:
            raise ValueError(f"Invalid mask_type: {self.mask_type}. Must be 'random' or 'non_obs_only'.")

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
            Input tensor of shape `(batch_size, n_hid_vars)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_hid_vars)`.
        """
        return nn.functional.linear(input, self.weight, self.bias)

class vanilla_cell(nn.Module):
    """
    Vanilla HCNN Cell.

    This class implements the Vanilla HCNN Cell for state-based predictions and state transitions.
    It supports the use of teacher forcing during training and provides methods to compute the 
    next state and predicted outputs.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    A : CustomLinear
        Linear transformation module for updating the hidden state.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]
        Performs a forward pass through the Vanilla HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """


    def __init__(self, n_obs: int, 
                 n_hid_vars: int , 
                 init_range: Tuple[float,float] = (-0.75,0.75)):

        super(vanilla_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Parameter initialization 
         
        self.A = CustomLinear(in_features =n_hid_vars, out_features =n_hid_vars , bias = False ,init_range = init_range)

        self.register_buffer(name = 'ConMat', tensor= torch.eye(n_obs, n_hid_vars), persistent = False)
        
        self.register_buffer(name = 'Ide', tensor= torch.eye(n_hid_vars ), persistent = False)

        # Select device
        self.device = self._get_default_device()
        
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


    def forward(self, state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Vanilla HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor | shape=(n_hid_vars,)
            The current state tensor  `s_{t}`is the HCNN state vector of shape (n_hid_vars,)
        teacher_forcing : bool
            Whether to use teacher forcing for the state transition:
            - True: Use the provided observation (`observation`) to guide the state transition.
            - False: Compute the next state based on the model's prediction.
        observation : Optional[torch.Tensor], default=None | shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) corresponds to the observable at time t. Required when 
            `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - expectation : torch.Tensor
                Predicted observation tensor (`y_hat`) |  shape (n_obs,).
            - next_state : torch.Tensor
                Updated state tensor (`s_{t+1}`) |  shape (n_hid_vars,).
            - delta_term : `y_hat - y_true` | shape (n_obs,) | Optional , can be None.

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
        """
    # Compute the expected output (y_hat)
        # expectation = torch.matmul(self.ConMat, state)

        expectation = torch.matmul(state , self.ConMat.T)

        # print("expectation requires_grad:", expectation.requires_grad)
        if teacher_forcing:

            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")

            # Compute the delta term (y_true - y_hat)
            delta_term = observation - expectation

            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(delta_term,self.ConMat)

            r_state = state - teach_forc
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, delta_term
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, None
        


        
class partial_teacher_forcing(nn.Dropout):
    """
    Implements dropout with scaling to enable partial teacher forcing.

    This layer applies dropout to the delta term (`y_hat - y_true`) during state transitions
    to simulate partial teacher forcing, allowing for a controlled level of noise or guidance.

    Attributes
    ----------
    p : float
        Dropout probability. The fraction of elements to drop.
    inplace : bool
        Whether to perform the operation in-place.

    Methods
    -------
    forward(input: torch.Tensor) -> torch.Tensor:
        Applies dropout with scaling to the input tensor.
    """
    def __init__(self, p: float = 0., inplace: bool = False):
        """
        Initializes the partial teacher forcing dropout layer.

        Parameters
        ----------
        p : float, optional (default=0.)
            Dropout probability. Defaults to 0 (no dropout).
        inplace : bool, optional (default=False)
            Whether to perform the operation in-place.
        """
        super(partial_teacher_forcing, self).__init__(p, inplace)

        self.p  =  p
        self.inplace = inplace


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies dropout to the input tensor with scaling.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor to which dropout will be applied.

        Returns
        -------
        torch.Tensor
            The scaled tensor with elements randomly dropped based on the dropout probability.
        """
        if not self.training or self.p == 0: #if no training or p = 0
            return input
        else:
            scaled_output = super().forward(input)
            return (1 - self.p) * scaled_output #applies prob dropout without inplace

        
        
class ptf_cell(nn.Module):

    """
    Partial Teacher Forcing HCNN Cell.

    This class implements the HCNN Cell with support for partial teacher forcing during training.
    It allows a controlled adjustment of the teacher forcing behavior using dropout probabilities, 
    which can simulate the effects of partial guidance in state transitions.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    A : CustomLinear
        Linear transformation module for updating the hidden state.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    ptf_dropout(prob: float) -> nn.Module:
        Returns a dropout module to apply partial teacher forcing.
    forward(state: torch.Tensor, teacher_forcing: bool, prob: float, 
            observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Performs a forward pass with optional partial teacher forcing during state transitions.
    """

    def __init__(self, n_obs: int, n_hid_vars: int, init_range: Tuple[float, float] = (-0.75, 0.75)):
        super(ptf_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Initialize layers and buffers
        self.A = CustomLinear(n_hid_vars, n_hid_vars, bias=False, init_range=init_range)
        self.register_buffer(name='ConMat', tensor=torch.eye(n_obs, n_hid_vars), persistent=False)
        self.register_buffer(name='Ide', tensor=torch.eye(n_hid_vars), persistent=False)

        # Select device
        self.device = self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def ptf_dropout(self, prob: float) -> nn.Module:

        """
        Creates a dropout module for applying partial teacher forcing.

        Parameters
        ----------
        prob : float
            Dropout probability for partial teacher forcing. Determines the fraction of elements in the
            delta term tensor (`y_hat - y_true`) that are randomly dropped (set to 0).

        Returns
        -------
        nn.Module
            A `partial_teacher_forcing` dropout module initialized with the given probability.
        """

        return partial_teacher_forcing(p=prob)

    def forward(self, state: torch.Tensor, teacher_forcing: bool, prob: Optional[float] = None,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Compute expected output (y_hat)
        """
        Forward pass of the Partial Teacher Forcing HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor, shape=(n_hid_vars,)
            The current state tensor (`s_t`) of the HCNN, with shape `(n_hid_vars,)`.
        teacher_forcing : bool
            Whether to apply teacher forcing:
            - True: Use the provided observation (`observation`) with optional dropout.
            - False: Compute the next state based solely on the model's prediction.
        prob : float
            Dropout probability for partial teacher forcing. Only used if `teacher_forcing` is True.
        observation : Optional[torch.Tensor], default=None, shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) for time step `t`.
            Required when `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - expectation : torch.Tensor, shape=(n_obs,)
                Predicted observation tensor (`y_hat`).
            - next_state : torch.Tensor, shape=(n_hid_vars,)
                Updated state tensor (`s_{t+1}`).
            - delta_term : Optional[torch.Tensor], shape=(n_obs,)
                Difference between the prediction and ground-truth (`y_hat - y_true`).
                Returns None when `teacher_forcing` is False.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the delta term is passed through partial dropout before
        being used to compute the next state.
        - Without teacher forcing, the next state is computed purely from the model's prediction.
        """
        expectation = torch.matmul(self.ConMat, state)

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            delta_term = observation - expectation
            partial_delta_term = self.ptf_dropout(prob)(delta_term)
            teach_forc = torch.matmul(self.ConMat.T, partial_delta_term)
            r_state = state - teach_forc
            next_state = self.A(torch.tanh(r_state))
            return expectation, next_state, delta_term, partial_delta_term
        else:
            r_state = torch.matmul(self.Ide, state)
            next_state = self.A(torch.tanh(r_state))
            return expectation, next_state, None, None


class DiagonalMatrix(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, init_diag: Optional[float] = 1.0):
        if in_features != out_features:
            raise ValueError("DiagonalMatrix requires in_features to be equal to out_features for symmetry.")
        super(DiagonalMatrix, self).__init__(in_features, out_features, bias=bias)

        # Initialize the weight matrix
        nn.init.constant_(self.weight, 0)  # Set all weights to zero
        if init_diag is not None:
            self.weight.data.fill_diagonal_(init_diag)  # User-specified or default value
        else:
            nn.init.uniform_(self.weight.data.fill_diagonal_(0), -1e-5, 1e-5)  # Random diagonal values


        # Pre-compute and store the diagonal mask
        self.register_buffer("mask", torch.eye(self.weight.shape[0], device=self.weight.device))

        # Register hook
        self.weight.register_hook(self._clamp_and_zero_out)


    def _clamp_and_zero_out(self, grad):

        """
        Zero out off-diagonal elements and clamp diagonal values
        Grad here, gradient_loss_wrt_weight.
        """
        with torch.no_grad():
            self.weight.data = torch.mul(self.weight.data ,self.mask)  # Retain only diagonal elements
            self.weight.data.clamp_(min=0., max=1)

        return grad * self.mask  # Mask gradients to zero out off-diagonal elements
    

    def forward(self, input, verbose: bool = False):
            if verbose:
                print(f"Diagonal weights: {torch.diag(self.weight)}")
            return nn.functional.linear(input, self.weight, self.bias)





class lstm_cell(nn.Module):
    """
    LSTM Formulation of the HCNN (LForm HCNN Cell).

    This class implements the LSTM-inspired formulation of the HCNN Cell for state-based predictions and state transitions.
    It incorporates a custom diagonal matrix transformation to enforce exponential embedding of the residual state.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    A : CustomLinear
        Linear transformation module for processing the non-linear residual state.
    D : DiagonalMatrix
        Diagonal matrix transformation module for controlling the residual state dynamics.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    _get_default_device() -> torch.device:
        Determines the default device for computations.
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the LSTM HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """

    def __init__(self, n_obs: int, n_hid_vars: int, init_range: Tuple[float, float] = (-0.75, 0.75)):
        super(lstm_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Initialize layers
        self.A = CustomLinear(n_hid_vars, n_hid_vars, bias=False, init_range=init_range)
        self.D = DiagonalMatrix(n_hid_vars, n_hid_vars, bias=False, init_diag=1.0)
        self.register_buffer(name='ConMat', tensor=torch.eye(n_obs, n_hid_vars), persistent=False)
        self.register_buffer(name='Ide', tensor=torch.eye(n_hid_vars), persistent=False)

        # Select device
        self.device = self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def forward(self, state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        """
        Forward pass of the LSTM HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor, shape=(n_hid_vars,)
            The current state tensor (`s_t`) of the HCNN, with shape `(n_hid_vars,)`.
        teacher_forcing : bool
            Whether to use teacher forcing for the state transition:
            - True: Use the provided observation (`observation`) to guide the state transition.
            - False: Compute the next state based on the model's prediction.
        observation : Optional[torch.Tensor], default=None, shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) for time step `t`.
            Required when `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - expectation : torch.Tensor, shape=(n_obs,)
                Predicted observation tensor (`y_hat`).
            - next_state : torch.Tensor, shape=(n_hid_vars,)
                Updated state tensor (`s_{t+1}`).
            - delta_term : Optional[torch.Tensor], shape=(n_obs,)
                Difference between the prediction and ground-truth (`y_hat - y_true`).
                Returns None when `teacher_forcing` is False.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the method uses `observation` to compute a correction term 
        (`delta_term`) for guiding the state transition.
        - If `teacher_forcing` is False, the state transition is based purely on the model's prediction.

        - State Transition Logic:
            - Compute the residual state (`r_state`).
            - Apply a non-linear transformation using the `CustomLinear` layer (`A`).
            - Add an adjusted residual term using the diagonal matrix (`D`).
        """

        
        # Compute expected output (y_hat)
        expectation = torch.matmul(self.ConMat, state)

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            delta_term = observation - expectation
            teach_forc = torch.matmul(self.ConMat.T, delta_term)
            r_state = state - teach_forc
            lstm_block = self.A(torch.tanh(r_state)) - r_state
            next_state = r_state + self.D(lstm_block)
            return expectation, next_state, delta_term
        else:
            r_state = torch.matmul(self.Ide, state)
            lstm_block = self.A(torch.tanh(r_state)) - r_state
            next_state = r_state + self.D(lstm_block)
            return expectation, next_state, None



class LargeSparse_cell(nn.Module):

    """
    Large Sparse HCNN Cell.

    This class implements the Large Sparse HCNN Cell for state-based predictions and state transitions.
    It supports the use of teacher forcing during training and includes a sparsity-enabled linear transformation
    for efficiently handling large state spaces.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    Sparse_A : CustomSparseLinear
        Sparse linear transformation module for updating the hidden state.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the Large Sparse HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """


    def __init__(self, n_obs: int, n_hid_vars: int, init_range: Tuple[float, float] = (-0.75, 0.75),
                 sparsity: float = 0.0, mask_type: str = "random", p: Optional[int] = None):
        

        """
        Initializes the Large Sparse HCNN Cell with a sparsity-enabled CustomSparseLinear.

        Parameters
        ----------
        n_obs : int
            Number of observed variables (output dimension).
        n_hid_vars : int
            Number of hidden variables (state dimension).
        init_range : Tuple[float, float], optional
            Range for uniform initialization of weights in CustomSparseLinear. Default is (-0.75, 0.75).
        sparsity : float, optional
            Proportion of weights to set to zero (random sparsity). Default is 0.0 (no sparsity).
        mask_type : str, optional
            Type of sparsity mask: "random" or "non_obs_only". Default is "random".
        p : int, optional
            Number of observable variables for "non_obs_only" mask type. Required if `mask_type` is "non_obs_only".
        """

        
        super(LargeSparse_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Initialize sparse transformation
        self.Sparse_A = CustomSparseLinear(n_hid_vars, bias=False, init_range=init_range,
                                           sparsity=sparsity, mask_type=mask_type, p=p)
        self.register_buffer(name='ConMat', tensor=torch.eye(n_obs, n_hid_vars), persistent=False)
        self.register_buffer(name='Ide', tensor=torch.eye(n_hid_vars), persistent=False)

        # Select device
        self.device = self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def forward(self, state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        """
        Forward pass of the Large Sparse HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor, shape=(n_hid_vars,)
            The current state tensor (`s_t`) of the HCNN, with shape `(n_hid_vars,)`.
        teacher_forcing : bool
            Whether to use teacher forcing for the state transition:
            - True: Use the provided observation (`observation`) to guide the state transition.
            - False: Compute the next state based on the model's prediction.
        observation : Optional[torch.Tensor], default=None, shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) for time step `t`.
            Required when `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - expectation : torch.Tensor, shape=(n_obs,)
                Predicted observation tensor (`y_hat`).
            - next_state : torch.Tensor, shape=(n_hid_vars,)
                Updated state tensor (`s_{t+1}`).
            - delta_term : Optional[torch.Tensor], shape=(n_obs,)
                Difference between the prediction and ground-truth (`y_hat - y_true`).
                Returns None when `teacher_forcing` is False.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the method uses `observation` to compute a correction term 
        (`delta_term`) for guiding the state transition.
        - If `teacher_forcing` is False, the state transition is based purely on the model's prediction.

        - State Transition Logic:
            - Compute the residual state (`r_state`).
            - Apply a non-linear transformation using the `CustomSparseLinear` layer (`Sparse_A`).
        """
        # Compute expected output (y_hat)
        expectation = torch.matmul(self.ConMat, state)

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            delta_term = observation - expectation
            teach_forc = torch.matmul(self.ConMat.T, delta_term)
            r_state = state - teach_forc
            next_state = self.Sparse_A(torch.tanh(r_state))
            return expectation, next_state, delta_term
        else:
            r_state = torch.matmul(self.Ide, state)
            next_state = self.Sparse_A(torch.tanh(r_state))
            return expectation, next_state, None
