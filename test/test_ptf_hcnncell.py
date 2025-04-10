

import pytest
import torch
from typing import Optional , Tuple , Literal
import torch.nn as nn
# import pytest
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


    r"""
    Partial Teacher Forcing HCNN Cell (HCNN-pTF).

    This class implements a HCNN Cell endowed with a partial teacher forcing mechanism during training.
    It performs a state-to-state mapping and produces outputs based on hidden states.
    It allows a controlled adjustment of the teacher forcing behavior using dropout probabilities.


    Parameters
    ----------
    `n_obs_var` : int
        Number of observed variables (i.e., the dimensionality of the observed state variables).

    `n_hid_vars` : int
        Number of hidden variables (i.e., the dimensionality of the hidden state variables).

    `init_range` : Tuple[float, float], optional
        Tuple specifying the range for uniform weight initialization in the `CustomLinear` module.
        Default is (-0.75, 0.75).


    Direct Attributes (from inputs)
    -------------------------------
    `n_obs_var` : int
        Stores the number of observed variables as a class attribute.

    `n_hid_vars` : int
        Stores the number of hidden variables as a class attribute.

    `init_range` : Tuple[float, float]
        Stores the initialization range for the `CustomLinear` module.


    Indirect Attributes (initialized internally)
    --------------------------------------------

    `n_state_vars` : int
        Total number of state variables (sum of hidden and observed variables).
        
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

    `ptf_dropout` : nn.Module
        A dropout module for applying partial teacher forcing.
        Initialized with the specified dropout probability.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, prob: Optional[float] = None,
           observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        Performs a forward pass through the Partial Teacher Forcing HCNN Cell.

        Returns 
        - the predicted observation (`expectation`), 
        - the next hidden state (`next_state`), and
        - the delta term (`y_hat - y_true`) which is the difference between the observation and expectation at time t
        if teacher forcing is True and None otherwise.
        - the partial delta term which is the difference between the expectation and the observation after applying dropout.
    """

    def __init__(self, n_obs_vars: int, n_hid_vars: int, init_range: Tuple[float, float] = (-0.75, 0.75)):

        super(ptf_cell, self).__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars

        self.init_range = init_range
        # Parameter initialization 
        
        self.A = CustomLinear(in_vars = self.n_state_vars, 
                            out_vars =self.n_state_vars , 
                            bias = False ,
                            init_range = self.init_range )


        self.register_buffer(name = 'ConMat', 
                            tensor= torch.eye(self.n_obs_vars, self.n_state_vars), 
                            persistent = False)
        

        self.register_buffer(name = 'Ide', 
                            tensor= torch.eye(self.n_state_vars ),
                            persistent = False)

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

    def forward(self, state: torch.Tensor, teacher_forcing: bool=False, prob: Optional[float] = 0.,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Compute expected output (y_hat)
        r"""
        Forward pass of the Partial Teacher Forcing HCNN Cell.

        Parameters
        ----------
        `state` : torch.Tensor | shape = (`n_state_vars`,)

            The current state tensor  `$\mathbf{s}_{t}$` is the HCNN state vector of shape (`n_state_vars`,)

        `teacher_forcing` : bool | default=False

            Whether to use teacher forcing for the state transition:

            - `True`: Use the provided observation (`observation`) to guide the state transition.
            - `False`: Compute the next state based on the model's prediction.

        `prob` : float, optional
            Dropout probability for partial teacher forcing. Only used if `teacher_forcing` is True.

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

            - delta_term :  torch.Tensor
             
                `y_hat - y_true` | shape (n_obs_var,) | Optional , can be None.

            - partial_delta_term :  torch.Tensor 
                
                `y_hat - y_true` after applying dropout | shape (n_obs_var,) | Optional , can be None.
        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the delta term is passed through partial dropout before
        being used to compute the next state.
        - Without teacher forcing, the next state is computed purely from the model's expectations.
        """
        expectation = torch.matmul(state , self.ConMat.T)

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")
            
            delta_term = observation - expectation
            
            partial_delta_term = self.ptf_dropout(prob)(delta_term)
            
            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(partial_delta_term,self.ConMat)
            
            r_state = state - teach_forc
            
            next_state = self.A(torch.tanh(r_state))
            
            return expectation, next_state, delta_term, partial_delta_term
        else:
            r_state = torch.matmul(state, self.Ide )
            next_state = self.A(torch.tanh(r_state))
            
            return expectation, next_state, None, None





def test_ptf_cell_initialization():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    assert cell.n_obs_vars == n_obs
    assert cell.n_hid_vars == n_hid_vars
    assert cell.ConMat.shape == (n_obs, n_obs + n_hid_vars)
    assert cell.Ide.shape == (n_obs + n_hid_vars, n_obs + n_hid_vars)
    assert isinstance(cell.A, CustomLinear)


def test_ptf_cell_device_assignment():
    cell = ptf_cell(5, 10)
    expected_device = cell._get_default_device()
    assert cell.device == expected_device


def test_ptf_cell_forward_no_dropout():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    observation = torch.randn(n_obs)

    y_hat, next_state, delta, delta_dropped = cell(state, teacher_forcing=True, prob=0.0, observation=observation)

    assert y_hat.shape == (n_obs,)
    assert next_state.shape == (n_obs + n_hid_vars,)
    assert delta.shape == (n_obs,)
    assert delta_dropped.shape == (n_obs,)
    assert torch.allclose(delta, observation - y_hat)
    assert torch.allclose(delta_dropped, delta)


def test_ptf_cell_partial_dropout_effect():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars)
    observation = torch.randn(n_obs)

    y_hat, _, delta, delta_dropped = cell(state, teacher_forcing=True, prob=1.0, observation=observation)

    assert torch.allclose(delta, observation - y_hat)
    assert torch.all(delta_dropped == 0), "With prob=1.0, all elements in delta should be dropped"


def test_ptf_cell_forward_no_teacher_forcing():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars)

    y_hat, next_state, delta, delta_dropped = cell(state, teacher_forcing=False)

    assert y_hat.shape == (n_obs,)
    assert next_state.shape == (n_obs + n_hid_vars,)
    assert delta is None
    assert delta_dropped is None


def test_ptf_cell_consistency_without_dropout():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars)
    observation = torch.randn(n_obs)

    out1 = cell(state, teacher_forcing=True, prob=0.0, observation=observation)
    out2 = cell(state, teacher_forcing=True, prob=0.0, observation=observation)

    for x, y in zip(out1, out2):
        if x is not None:
            assert torch.allclose(x, y), "Inconsistent output for identical input when dropout disabled"


def test_ptf_cell_error_on_missing_observation():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars)
    with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
        cell(state, teacher_forcing=True)


def test_ptf_cell_no_gradient_flow():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    observation = torch.randn(n_obs)

    y_hat, next_state, *_ = cell(state, teacher_forcing=True, prob=0.0, observation=observation)

    loss = MSELoss()(y_hat, observation)
    loss.backward()

    assert state.grad is not None, "No gradients flowing to input state as requires_grad=False"
    assert cell.A.weight.grad is  None, "No gradients should be flowing to A weights as it is not part of the computational graph"

def test_ptf_cell_gradient_flow():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    observation = torch.randn(n_obs)

    y_hat, next_state, *_ = cell(state, teacher_forcing=True, prob=0.0, observation=observation)

    loss = MSELoss()(next_state, torch.randn(n_obs + n_hid_vars))
    loss.backward()

    assert state.grad is not None, "gradients should flow into to input state"
    assert cell.A.weight.grad is not None, "gradients should be flowing to A weights as it is part of the computational graph"



def test_ptf_cell_sequence_recurrence():
    T, n_obs, n_hid_vars = 4, 6, 8
    cell = ptf_cell(n_obs, n_hid_vars)
    loss_fn = MSELoss()

    obs_seq = torch.randn(T, n_obs)
    s0 = torch.randn(n_obs + n_hid_vars, requires_grad=True)

    states, yhats = [s0], []
    for t in range(T - 1):
        y_hat, next_state, *_ = cell(states[-1], teacher_forcing=True, prob=0.0, observation=obs_seq[t])
        yhats.append(y_hat)
        states.append(next_state)

    yhats.append(torch.matmul(states[-1], cell.ConMat.T))
    loss = loss_fn(torch.stack(yhats), obs_seq)

    loss.backward()

    assert s0.grad is not None
    assert cell.A.weight.grad is not None
    assert loss.item() > 0


def test_ptf_cell_dropout_behavior_extremes():
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)
    state = torch.randn(n_obs + n_hid_vars)
    observation = torch.randn(n_obs)

    # No dropout
    _, _, delta, partial = cell(state, teacher_forcing=True, prob=0.0, observation=observation)
    assert torch.allclose(partial, delta), "Expected no dropout applied"

    # Full dropout
    _, _, delta, partial = cell(state, teacher_forcing=True, prob=1.0, observation=observation)
    assert torch.all(partial == 0), "Expected all entries to be dropped with prob=1.0"
