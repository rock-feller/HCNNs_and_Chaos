import pytest
import torch , os
from typing import Optional , Tuple , Literal
import torch.nn as nn
# import pytest
from torch.nn import MSELoss
from torch.optim import Adam
from collections import namedtuple


# Named tuple for the output of the Vanilla HCNN model
VanillaHCNNForwardOutput = namedtuple(
    "VanillaHCNNForwardOutput", 
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)


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




class Vanilla_Model(nn.Module):
    """
    Wrapper Model for the Vanilla HCNN Cell.

    This model wraps the `vanilla_cell` to provide a batched, multi-step forward processing and
    forecasting interface. It supports teacher forcing during training and auto-regressive
    forecasting during inference.

    Parameters
    ----------
    `n_obs_vars` : int
        Number of observable variables (i.e., output dimensionality).

    `n_hid_vars` : int
        Number of hidden (latent) variables in the state.

    `s0_nature` : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `s0`.
        - 'zeros_': Initialize as a zero vector.
        - 'random_': Initialize uniformly using `init_range`.

    `train_s0` : bool
        If True, the initial state `s0` is a trainable parameter.

    `batch_size` : int, optional
        Number of sequences to process in parallel. Default is 1.

    `init_range` : Tuple[float, float], optional
        Range for uniform initialization when `s0_nature='random_'`. Also passed to the internal cell.

        
    Direct Attributes (from inputs)
    -------------------------------
    `n_obs_vars` : int
        Number of observed variables.

    `n_hid_vars` : int
        Number of hidden variables.

    `s0_nature` : str
        Initial state setup strategy ('zeros_' or 'random_').

    `train_s0` : bool
        Whether the initial hidden state `s0` is trainable.

    `batch_size` : int
        Number of parallel input sequences to process.

    `init_range` : Tuple[float, float]
        Initialization range for random initialization.

    Indirect Attributes (initialized internally)
    --------------------------------------------
    `cell` : vanilla_cell
        The core recurrent computation unit of the model.

    s0 : torch.nn.Parameter
        The initial hidden state tensor of shape (1, n_hid_vars).
        Repeated across the batch in the forward pass.

    `name` : str
        A unique model name based on initialization and training configuration.

    `device` : torch.device
        Computation device (CUDA, MPS, or CPU) used by the model.

    Methods
    -------
    forward(data_window: torch.Tensor,
            forecast_horizon: Optional[int] = None ) -> VanillaHCNNForwardOutput
        Performs forward pass for the entire sequence, including optional future forecasting.

    initial_hidden_state() -> nn.Parameter:
        Returns the learnable initial state vector.

    save_checkpoint(epoch: int, loss: float, optimizer: torch.optim.Optimizer) -> None:
        Saves a checkpoint of the model and optimizer state.

    load_checkpoint(checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
        Loads a saved checkpoint and optionally restores the optimizer state.
    """


    def __init__(self, n_obs_vars: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, batch_size: int = 1,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        
        super(Vanilla_Model, self).__init__()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars
        
        self.s0_nature = s0_nature
        self.train_s0 = train_s0

        self.batch_size = batch_size
        self.init_range = init_range

        self.cell = vanilla_cell(n_obs_vars = n_obs_vars, 
                                 n_hid_vars= n_hid_vars, 
                                 init_range= init_range)
        
        self.device = self.cell._get_default_device()
        self.name = self._generate_model_name()



        if self.s0_nature.lower() == "zeros_":
            s0 = torch.zeros(1, self.n_state_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            s0 = torch.empty(1, self.n_state_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `h0` a trainable parameter (single vector, not repeated for batch size)
        self.s0 = nn.Parameter(s0, requires_grad=self.train_s0)

    def _generate_model_name(self) -> str:
        """Generates a unique model name based on configuration."""

        name = f"VanillaModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
        if self.s0_nature == "random_":
            name += f"_randInit{self.init_range[0]}to{self.init_range[1]}"
        else:
            name += "_zeroInit"
        if self.train_s0:
            name += "_trainableS0"
        return name

    def initial_hidden_state(self) -> nn.Parameter:
        """
        Return the trainable initial hidden state.
        """
        return self.s0
    

    def forward(self, data_window: torch.Tensor,
                forecast_horizon: Optional[int] = None) -> VanillaHCNNForwardOutput:
        """
        Executes a forward pass through the Vanilla HCNN model over a sequence of observations.
        Supports both teacher-forced training and optional auto-regressive forecasting.

        Parameters
        ----------
        `data_window` : torch.Tensor
            Observed input sequence of shape (batch_size, sequence_length, n_obs_vars).
            Represents a batch of multivariate time series.

        `forecast_horizon` : Optional[int], default=None
            Number of future steps to forecast beyond the observed `data_window`.
            If None, no forecasting is performed. If specified, the model enters
            auto-regressive prediction mode for `forecast_horizon` steps.

        Returns
        -------
        VanillaHCNNForwardOutput
            A namedtuple containing the following tensors:

            - `expectations` : torch.Tensor
                Predicted observations for each time step in the input window.
                Shape: (batch_size, sequence_length, n_obs_vars)

            - `states` : torch.Tensor
                Hidden states for each time step in the input window.
                Shape: (batch_size, sequence_length, n_hid_vars)

            - `delta_terms` : torch.Tensor
                Differences between ground-truth and predicted observations (y_true - y_hat).
                acros the input window.
                Shape: (batch_size, sequence_length, n_obs_vars)

            - `forecasts` : Optional[torch.Tensor]
                Predicted observations for `forecast_horizon` future steps.
                Only returned if `forecast_horizon` is specified.
                Shape: (batch_size, forecast_horizon, n_obs_vars)

            - `future_states` : Optional[torch.Tensor]
                Hidden states associated with the predicted future steps.
                Shape: (batch_size, forecast_horizon, n_hid_vars)

        Notes
        -----
        - During training, teacher forcing is used on the `data_window` to guide state updates.
        - During forecasting, the last state from the input sequence is propagated without teacher forcing.
        - The initial hidden state `s0` is shared across all sequences and may be learned or fixed depending
          on `train_s0`.
        """

        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_state_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.s0
        if seq_length > 1:

            # Process observed data
            for t in range(seq_length - 1):
                expectation, next_state, delta_term = self.cell(
                    state=states[:, t, :],
                    teacher_forcing=True,
                    observation=data_window[:, t, :]
                )
                expectations[:, t, :] = expectation
                states[:, t + 1, :] = next_state
                delta_terms[:, t, :] = delta_term

            # Final observed time step
            # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            # expectations[:, seq_length - 1, :] = last_y_hat

            last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
            

            expectations[:, seq_length - 1, :] = last_y_hat
            delta_terms[:, seq_length - 1, :] = last_delta_term
            # partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
        else:

            last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)


            expectations[:, seq_length - 1, :] = last_y_hat
            delta_terms[:, seq_length - 1, :] = last_delta_term
        # Process observed data
        # for t in range(seq_length - 1):
        #     expectation, next_state, delta_term = self.cell(
        #         state=states[:, t, :],
        #         teacher_forcing=True,
        #         observation=data_window[:, t, :]
        #     )
        #     expectations[:, t, :] = expectation
        #     states[:, t + 1, :] = next_state
        #     delta_terms[:, t, :] = delta_term

        # # Final observed time step
        # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
        # expectations[:, seq_length - 1, :] = last_y_hat

        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs_vars, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_state_vars, device=self.device)
            teach_forc = torch.matmul(last_delta_term,self.cell.ConMat)

            r_state = states[:, 0, :] - teach_forc
            next_state = self.cell.A(torch.tanh(r_state))

            with torch.no_grad():
                # Use the last observed state as the starting point
                future_states[:, 0, :] = states[:, seq_length - 1, :]

                # Forecast future steps
                for t in range(1, forecast_horizon):
                    forecast, next_state, _ = self.cell(
                        state=future_states[:, t - 1, :],
                        teacher_forcing=False
                    )
                    forecasts[:, t - 1, :] = forecast
                    future_states[:, t, :] = next_state

            forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)

        return VanillaHCNNForwardOutput( expectations=expectations, states=states, delta_terms=delta_terms, forecasts=forecasts, future_states=future_states )


    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer):
        """Saves model checkpoint."""
        checkpoint_path = f"checkpoints/{self.name}_epoch{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)  # Ensure the directory exists
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Loads model checkpoint."""
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])  # Load model parameters
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Load optimizer state
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            print(f"Checkpoint loaded from {checkpoint_path}. Epoch: {epoch}, Loss: {loss}")
            return epoch, loss
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


# Assuming Vanilla_Model and VanillaHCNNForwardOutput are imported correctly

def test_model_initialization_zero():
    """Test Vanilla_Model initialization with zero state vector."""
    model = Vanilla_Model(n_obs_vars=5, n_hid_vars=10, s0_nature='zeros_', train_s0=False)
    assert model.s0.shape == (1, 15)
    assert torch.allclose(model.s0, torch.zeros_like(model.s0))
    assert model.s0.requires_grad is False

def test_model_initialization_random():
    """Test Vanilla_Model initialization with random state vector."""
    model = Vanilla_Model(n_obs_vars=5, n_hid_vars=10, s0_nature='random_', train_s0=True)
    low, high = model.init_range
    assert (model.s0 >= low).all() and (model.s0 <= high).all()
    assert model.s0.requires_grad is True

def test_forward_pass_shape():
    """Test output shapes of forward pass without forecasting."""
    model = Vanilla_Model(n_obs_vars=3, n_hid_vars=7, s0_nature='zeros_', train_s0=True)
    batch_size, seq_len = 2, 4
    data = torch.randn(batch_size, seq_len, 3)
    output = model(data)

    assert output.expectations.shape == (batch_size, seq_len, 3)
    assert output.states.shape == (batch_size, seq_len, 10)
    assert output.delta_terms.shape == (batch_size, seq_len, 3)
    assert output.forecasts is None
    assert output.future_states is None

def test_forward_with_forecasting():
    """Test forward pass with forecast horizon."""
    model = Vanilla_Model(n_obs_vars=4, n_hid_vars=6, s0_nature='random_', train_s0=True)
    batch_size, seq_len, horizon = 3, 5, 2
    data = torch.randn(batch_size, seq_len, 4)
    output = model(data, forecast_horizon=horizon)

    assert output.forecasts is not None
    assert output.future_states is not None
    assert output.forecasts.shape == (batch_size, horizon, 4)
    assert output.future_states.shape == (batch_size, horizon, 10)

def test_model_name_generation():
    """Test automatic name generation for model configuration."""
    model = Vanilla_Model(n_obs_vars=2, n_hid_vars=3, s0_nature='zeros_', train_s0=False)
    assert "VanillaModel_obs2_hid3_zeroInit" in model.name

def test_invalid_s0_nature_raises():
    """Test that invalid s0_nature raises an exception."""
    with pytest.raises(ValueError, match="s0_nature must be either 'zeros_' or 'random_'"):
        _ = Vanilla_Model(n_obs_vars=2, n_hid_vars=3, s0_nature='invalid', train_s0=True)

def test_initial_hidden_state_reference():
    """Test that the initial hidden state is returned correctly."""
    model = Vanilla_Model(n_obs_vars=2, n_hid_vars=5, s0_nature='zeros_', train_s0=True)
    assert torch.equal(model.initial_hidden_state(), model.s0)

# def test_save_and_load_checkpoint(tmp_path):
#     """Test saving and loading a model checkpoint."""
#     model = Vanilla_Model(n_obs_vars=2, n_hid_vars=4, s0_nature='zeros_', train_s0=True)
#     optimizer = Adam(model.parameters(), lr=0.01)

#     dummy_loss = 1.23
#     dummy_epoch = 5

#     save_path = tmp_path / f"{model.name}_epoch{dummy_epoch}.pth"
#     model.save_checkpoint(dummy_epoch, dummy_loss, optimizer)

    # Simulate reloading model
    # loaded_model = Vanilla_Model(n_obs_vars=2, n_hid_vars=4, s0_nature='zeros_', train_s0=True)
    # loaded_epoch, loaded_loss = loaded_model.load_checkpoint(str(save_path), optimizer)

    # assert loaded_epoch == dummy_epoch
    # assert loaded_loss == dummy_loss

def test_autoregressive_consistency():
    """Check forecast prediction without gradient tracking."""
    model = Vanilla_Model(n_obs_vars=3, n_hid_vars=6, s0_nature="zeros_", train_s0=False)
    data = torch.randn(2, 3, 3)
    output = model(data, forecast_horizon=2)
    assert not output.forecasts.requires_grad, "Forecasts should not track gradients"
    assert not output.future_states.requires_grad, "Future states should not track gradients"
