

import pytest
from torch.nn import MSELoss
import torch
from typing import Optional , Tuple , Literal
import torch.nn as nn
# import pytest


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




class vanilla_cell(nn.Module):
   
    """
        Vanilla HCNN Cell.

        This class implements a vanilla version of the Historical Consistent Neural Network (HCNN) Cell.
        It performs a state-to-state mapping and produces outputs based on hidden states.
        It is designed to support teacher forcing during training.

        Parameters
        ----------
        `n_obs_vars` : int
            Number of observed variables (i.e., the dimensionality of the observed state variables).

        `n_hid_vars` : int
            Number of hidden variables (i.e., the dimensionality of the hidden state variables).

        `init_range` : Tuple[float, float], optional
            Tuple specifying the range for uniform weight initialization in the `CustomLinear` module.
            Default is (-0.75, 0.75).

        Direct Attributes (from inputs)
        -------------------------------
        `n_obs_vars` : int
            Stores the number of observed variables as a class attribute.

        `n_hid_vars` : int
            Stores the number of hidden variables as a class attribute.

        `init_range` : Tuple[float, float]
            Stores the initialization range for the `CustomLinear` module.

        Indirect Attributes (initialized internally)
        --------------------------------------------

        `n_state_vars` : int
            Total number of state variables (sum of hidden and observed variables).
            
        `A` : `CustomLinear`
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

        Methods
        -------
        forward(state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor |None]
            Performs a forward pass through the Vanilla HCNN Cell.

            Returns 
            - the predicted observation (`expectation`) 
            - the next hidden state (`next_state`) and
            - the delta_term (`y_hat - y_true`) which is the difference between the observation and expectation at time t
             if teacher forcing is True and None otherwise .
        
    """

    def __init__(self, n_obs_vars: int,  n_hid_vars: int , 
            init_range: Tuple[float,float] = (-0.75,0.75)):

    
        super(vanilla_cell, self).__init__()
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


    def forward(self, state: torch.Tensor, teacher_forcing: bool=False,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
            r_state = torch.matmul( state,self.Ide)
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, None
        



# from hcnn.cells import vanilla_cell, CustomLinear  # Adjust import path as needed

def test_initialization():
    """Test if the VanillaHCNNCell initializes correctly."""
    n_obs_vars, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars)
    assert cell.n_obs_vars == n_obs_vars
    assert cell.n_hid_vars == n_hid_vars
    assert cell.ConMat.shape == (n_obs_vars, n_obs_vars + n_hid_vars)
    assert cell.Ide.shape == (n_obs_vars + n_hid_vars, n_obs_vars + n_hid_vars)
    assert isinstance(cell.A, CustomLinear)

def test_device_assignment():
    """Test if the device is correctly selected."""
    cell = vanilla_cell(n_obs_vars=5, n_hid_vars=10)
    expected = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    assert cell.device == expected

def test_forward_without_teacher_forcing():
    """Check output shapes and behavior when teacher forcing is off."""
    n_obs_vars, n_hid_vars = 6, 12
    cell = vanilla_cell(n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars)
    state = torch.randn(n_obs_vars + n_hid_vars)

    expectation, next_state, delta = cell(state, teacher_forcing=False)
    assert expectation.shape == (n_obs_vars,)
    assert next_state.shape == (n_obs_vars + n_hid_vars,)
    assert delta is None

def test_forward_with_teacher_forcing():
    """Check outputs when teacher forcing is active."""
    n_obs_vars, n_hid_vars = 6, 12
    cell = vanilla_cell(n_obs_vars=n_obs_vars, n_hid_vars=n_hid_vars)
    state = torch.randn(n_obs_vars + n_hid_vars)
    obs = torch.randn(n_obs_vars)

    expectation, next_state, delta = cell(state, teacher_forcing=True, observation=obs)
    assert expectation.shape == (n_obs_vars,)
    assert next_state.shape == (n_obs_vars + n_hid_vars,)
    assert delta.shape == (n_obs_vars,)

def test_forward_with_teacher_forcing_missing_obs():
    """Ensure ValueError is raised when teacher forcing is enabled without observation."""
    cell = vanilla_cell(n_obs_vars=4, n_hid_vars=8)
    state = torch.randn(12)

    with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
        cell(state, teacher_forcing=True)

def test_consistent_outputs():
    """Same input to cell should yield identical output if dropout not involved."""
    n_obs, n_hid_vars = 4, 6
    cell = vanilla_cell(n_obs_vars=n_obs, n_hid_vars=n_hid_vars)
    state = torch.randn(n_obs + n_hid_vars)
    obs = torch.randn(n_obs)

    out1 = cell(state, teacher_forcing=True, observation=obs)
    out2 = cell(state, teacher_forcing=True, observation=obs)

    for x, y in zip(out1, out2):
        if x is not None:
            assert torch.allclose(x, y, atol=1e-6)

def test_nogradient_flow_single_step():
    """Ensure gradients flow through the A weights during backprop."""
    n_obs, n_hid_vars = 3, 5
    cell = vanilla_cell(n_obs_vars=n_obs, n_hid_vars=n_hid_vars)
    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    obs = torch.randn(n_obs)
    loss_fn = MSELoss()
    target = torch.randn_like(state)

    _, next_state, _ = cell(state, teacher_forcing=True, observation=obs)
    loss = loss_fn(next_state, target)
    loss.backward()

    assert state.grad is not None, "gradients should flow into to input state"
    assert cell.A.weight.grad is not None, "gradients should be flowing to A weights as it is part of the computational graph"

def test_gradient_flow_single_step():
    """Ensure no gradients flow through the A weights  during backprop."""
    n_obs, n_hid_vars = 3, 5
    cell = vanilla_cell(n_obs_vars=n_obs, n_hid_vars=n_hid_vars)
    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    obs = torch.randn(n_obs)
    loss_fn = MSELoss()
    target = torch.randn_like(state)

    expectation, next_state, _ = cell(state, teacher_forcing=True, observation=obs)
    loss = loss_fn(expectation, obs)
    loss.backward()

    assert state.grad is not None, "gradients should flow into to input state"
    assert cell.A.weight.grad is  None, "gradients should not be flowing to A weights as it is part of the computational graph"

def test_recurrent_usage():
    """Run the vanilla cell over a sequence of steps and backpropagate."""
    T, n_obs, n_hid_vars = 5, 3, 5
    cell = vanilla_cell(n_obs_vars=n_obs, n_hid_vars=n_hid_vars)
    observations = torch.randn(T, n_obs)
    state = torch.randn(n_obs + n_hid_vars, requires_grad=True)
    loss_fn = MSELoss()

    states = [state]
    yhats = []

    for t in range(T - 1):
        yhat, next_state, _ = cell(states[-1], teacher_forcing=True, observation=observations[t])
        yhats.append(yhat)
        states.append(next_state)

    last_yhat = torch.matmul(states[-1], cell.ConMat.T)
    yhats.append(last_yhat)
    yhats_tensor = torch.stack(yhats)

    loss = loss_fn(yhats_tensor, observations)
    loss.backward()

    assert state.grad is not None
    assert cell.A.weight.grad is not None
    assert loss.item() > 0
