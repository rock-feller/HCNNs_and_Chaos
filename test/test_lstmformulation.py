import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import MSELoss
# import unittest
import torch
import pytest
from torch.nn import MSELoss




class CustomLinear(nn.Linear):
    """
    Custom Linear layer with user-defined weight initialization.

    This class extends PyTorch's `nn.Linear` module to allow users to specify a custom
    range for uniform weight initialization. It is primarily used in HCNN
    for mapping between variables (`state-to-state`,  `external-to-state`).

    Parameters
    ----------
    in_vars : int
        Number of input variables. (i.e., the dimensionality of the input variables).
        In the context of the `state-to-state` mapping, this is the total number of
        state variables (including both observed and hidden variables).

        In the context of the `external-to-state` mapping, this is the number of
        external variables (i.e., the dimensionality of the external variables).
        This is referred to as `in_features` in the PyTorch `nn.Linear` module.

    out_vars : int
        Number of output variables. i.e., the dimensionality of the output variables).
        In the context of the `state-to-state` mapping, this is the total number of
        state variables (including both observed and hidden variables).

        In the context of the `external-to-state` mapping, this is the number of
        state variables (including both observed and hidden variables).
        This is referred to as `out_features` in the PyTorch `nn.Linear` module.

    bias : bool, optional
        Whether to include a bias term in the linear transformation. Default is False.

    init_range : Tuple[float, float], optional
        Tuple specifying the range (min, max) for uniform initialization of weights
        and optionally bias. Default is (-0.75, 0.75).

        
    Direct Attributes (from inputs)
    -------------------------------
    in_vars : int
        Number of input variables (alias for `in_features`). 


    out_vars : int
        Number of output variables (alias for `out_features`).
    
    bias : bool
        Whether to include a bias term in the linear transformation.

    init_range : Tuple[float, float]
        Range used for initializing the weights (and bias, if enabled). 
        

    Indirect Attributes (initialized internally)
    --------------------------------------------
    weight : torch.Tensor
        Weight matrix of shape `(out_features, in_features)` initialized uniformly 
        within the specified `init_range`.

    bias : torch.Tensor or None
        Optional bias vector of shape `(out_features,)` initialized uniformly within 
        `init_range`, if `bias=True`.

    Notes
    -----
    This class ensures consistent initialization when used in architectures that are 
    sensitive to the starting parameter values for the weights matrix and bias, such as HCNNs.
    """

    def __init__(self, in_vars: int, out_vars: int, bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        
        super(CustomLinear, self).__init__(in_features= in_vars, out_features=out_vars, bias=bias)

        self.in_features = in_vars
        self.out_features = out_vars

        self.init_range = init_range

        self.device = self._get_default_device()
        # Properly initialize weights and biases
        nn.init.uniform_(self.weight.data, self.init_range[0], self.init_range[1])
        if bias:
            nn.init.uniform_(self.bias.data, self.init_range[0], self.init_range[1])


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





class lstm_cell(nn.Module):

    r''''
    LSTM Formulation of the HCNN Cell (HCNN-LForm Cell).

    This module implements an LSTM-inspired variant of the Historical Consistent Neural Network (HCNN) cell.
    It performs a non-linear residual update using a learnable diagonal matrix to regulate the embedding of 
    residuals in the hidden state space. This structure is designed to support both autonomous dynamics and 
    teacher-forced based training.

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., the output dimensionality of the system).

    n_hid_vars : int
        Number of hidden variables (i.e., the internal state dimensionality).

    init_range : Tuple[float, float], optional
        Range for uniform initialization of the `CustomLinear` weight matrix. Default is (-0.75, 0.75).

    Direct Attributes (from inputs)
    -------------------------------
    n_obs_vars : int
        Stores the number of observed variables as a class attribute.

    n_hid_vars : int
        Stores the number of hidden variables as a class attribute.

    init_range : Tuple[float, float]
        Initialization range used for weights in the `CustomLinear` layer.

    Indirect Attributes (initialized internally)
    --------------------------------------------

    `A` : CustomLinear
        A linear transformation module that updates the hidden state.
        Configured with no bias and initialized using the provided `init_range`.

    `ConMat` : torch.Tensor
        A readout matrix that maps hidden states to observed outputs.
        Initialized as horizontal concatenation of an identity matrix of shape (`n_obs_var`, `n_obs_var`).
        and a zero matrix of  shape (`n_obs_var`, `n_state_vars`). 
        Registered as a non-persistent buffer.

    `Ide` : torch.Tensor
        Identity matrix used for internal computations on the hidden state.
        Shape is (`n_state_vars`, `n_state_vars`). Registered as a non-persistent buffer.

    `device` : torch.device
        The device (CPU, CUDA, or MPS) on which the model operates.
        Automatically detected using `_get_default_device`.

    `D` : DiagonalMatrix
        A learnable diagonal transformation used to regulate between memory conservation and computation dynamics.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool,
            observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        Executes a forward pass of the HCNN-LForm cell.
        Computes model output and state transitions using teacher forcing or self-propagation.

    _get_default_device() -> torch.device
        Detects the default computation device available.

    Notes
    -----
    - This formulation introduces LSTM-like dynamics into the HCNN by combining:
        1. Nonlinear transformations of the residual state (`A`),
        2. Diagonal modulation (`D`) of the  memory conservation (through the re-fit state vector $\mathbf{r}_{t}$
          and the memory computation of of the hidden state (through A \times \tanhh(\mathbf{S}_{t})) 

    - When `teacher_forcing` is enabled, the external observation is used to correct the internal state
      through a delta projection (`delta_term`).

    - If disabled, the cell performs autonomous prediction using the hidden state alone.

    Methods
    -------
    _get_default_device() -> torch.device:
        Determines the default device for computations.
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the LSTM HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    '''

    def __init__(self, n_obs_vars: int, n_hid_vars: int, 
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 init_diag: float = 1.0):
        
        super(lstm_cell, self).__init__()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.init_diag =  init_diag
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars

        # Initialize layers
        self.A = CustomLinear( in_vars = self.n_state_vars ,
                              out_vars= self.n_state_vars ,
                                bias=False, 
                                init_range=init_range)
        
        self.D = DiagonalMatrix(n_state_vars = self.n_state_vars ,
                                  bias=False, init_diag=self.init_diag)

        self.register_buffer(name='ConMat', tensor=torch.eye(self.n_obs_vars, self.n_state_vars), persistent=False)
        self.register_buffer(name='Ide', tensor=torch.eye(self.n_state_vars), persistent=False)


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
            
            lstm_block = self.A(torch.tanh(r_state)) - r_state
            
            next_state = r_state + self.D(lstm_block)
            
            return expectation, next_state, delta_term
        
        else:
            r_state = torch.matmul( state,self.Ide)

            lstm_block = self.A(torch.tanh(r_state)) - r_state

            next_state = r_state + self.D(lstm_block)

            return expectation, next_state, None


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




def test_lstm_cell_initialization():
    n_obs, n_hid = 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    assert cell.n_obs_vars == n_obs
    assert cell.n_hid_vars == n_hid
    assert cell.A.weight.shape == (n_obs + n_hid, n_obs + n_hid)
    assert cell.D.weight.shape == (n_obs + n_hid, n_obs + n_hid)
    assert torch.allclose(cell.D.weight.data, torch.diag(torch.diag(cell.D.weight.data)))
    assert (cell.ConMat.shape[0] == n_obs and cell.ConMat.shape[1] == n_obs + n_hid)


def test_lstm_cell_forward_no_teacher_forcing():
    n_obs, n_hid = 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid)
    y_hat, next_state, delta = cell(state, teacher_forcing=False)
    assert y_hat.shape == (n_obs,)
    assert next_state.shape == (n_obs + n_hid,)
    assert delta is None


def test_lstm_cell_forward_with_teacher_forcing():
    n_obs, n_hid = 5, 7
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid)
    obs = torch.randn(n_obs)
    y_hat, next_state, delta = cell(state, teacher_forcing=True, observation=obs)
    assert y_hat.shape == (n_obs,)
    assert next_state.shape == (n_obs + n_hid,)
    assert delta.shape == (n_obs,)
    assert torch.allclose(delta, obs - y_hat)


def test_lstm_cell_missing_observation_error():
    n_obs, n_hid = 3, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid)
    with pytest.raises(ValueError):
        cell(state, teacher_forcing=True)


def test_lstm_cell_nogradient_flow():
    n_obs, n_hid = 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid, requires_grad=True)
    obs = torch.randn(n_obs)
    y_hat, next_state, delta = cell(state, teacher_forcing=True, observation=obs)
    loss = MSELoss()(y_hat, obs)
    loss.backward()
    assert state.grad is not None
    assert cell.A.weight.grad is  None, "Gradient should not flow through A"
    assert cell.D.weight.grad is  None, "Gradient should not flow through D"


def test_lstm_cell_gradient_flow():
    n_obs, n_hid = 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid, requires_grad=True)
    obs = torch.randn(n_obs)
    y_hat, next_state, delta = cell(state, teacher_forcing=True, observation=obs)
    loss = MSELoss()(next_state, torch.randn(n_obs + n_hid))
    loss.backward()
    assert state.grad is not None
    assert cell.A.weight.grad is not None, "Gradient should  flow through A"
    assert cell.D.weight.grad is not None, "Gradient should flow through D"



def test_lstm_cell_diagonal_enforcement():
    n_obs, n_hid = 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid, requires_grad=True)
    obs = torch.randn(n_obs)
    y_hat, _, _ = cell(state, teacher_forcing=True, observation=obs)
    loss = MSELoss()(y_hat, obs)
    loss.backward()
    D = cell.D.weight.data
    off_diag = D - torch.diag(torch.diag(D))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-5)
    diag = torch.diag(D)
    assert torch.all((0.0 <= diag) & (diag <= 1.0))


def test_lstm_cell_recurrent():
    T, n_obs, n_hid = 5, 4, 6
    cell = lstm_cell(n_obs_vars=n_obs, n_hid_vars=n_hid)
    state = torch.randn(n_obs + n_hid, requires_grad=True)
    targets = torch.randn(T, n_obs)
    loss_fct = MSELoss()

    states = [state]
    yhats = []
    for t in range(T - 1):
        y_hat, next_state, _ = cell(states[t], teacher_forcing=True, observation=targets[t])
        yhats.append(y_hat)
        states.append(next_state)
    yhats.append(torch.matmul(cell.ConMat, states[-1]))
    yhats = torch.stack(yhats)

    loss = loss_fct(yhats, targets)
    loss.backward()

    assert state.grad is not None
    assert cell.A.weight.grad is not None
    assert cell.D.weight.grad is not None
    assert yhats.shape == (T, n_obs)
