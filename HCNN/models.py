import os, glob
import torch
from torch import nn
from typing import Literal, Tuple, Optional
# from torch.utils.tensorboard import SummaryWriter

from .modules import vanilla_cell , ptf_cell, lstm_cell, LargeSparse_cell, CustomLinear


from collections import namedtuple


# Named tuple for the output of the Vanilla HCNN model
VanillaHCNNForwardOutput = namedtuple(
    "VanillaHCNNForwardOutput", 
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)

# Named tuple for the output of the HCNN-pTF model
HCNNpTFForwardOutput = namedtuple(
    "HCNNpTFForwardOutput",
    ["expectations", "states", "delta_terms", "partial_delta_terms", "forecasts", "future_states"]
)

# Named tuple for the output of the HCNN-LForm model

HCNNLFormForwardOutput = namedtuple(
    "HCNNLFormForwardOutput",
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)

# Named tuple for the output of the HCNN-LSpa model

HCNNLSpaForwardOutput = namedtuple(
    "SparseHCNNForwardOutput",
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)


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

    `n_ext_vars` : int
        Number of external variables (optional). 

    `s0_nature` : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `s0`.
        - 'zeros_': Initialize as a zero vector.
        - 'random_': Initialize uniformly using `init_range`.

    `train_s0` : bool
        If True, the initial state `s0` is a trainable parameter.


    `init_range` : Tuple[float, float], optional
        Range for uniform initialization when `s0_nature='random_'`. Also passed to the internal cell.

        
    Direct Attributes (from inputs)
    -------------------------------
    `n_obs_vars` : int
        Number of observed variables.

    `n_hid_vars` : int
        Number of hidden variables.
    
    `n_ext_vars` : int
        Number of external variables (optional). 

    `s0_nature` : str
        Initial state setup strategy ('zeros_' or 'random_').

    `train_s0` : bool
        Whether the initial hidden state `s0` is trainable.


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
                 train_s0: bool, 
                 init_range: Tuple[float, float] = (-0.75, 0.75), 
                 n_ext_vars : Optional[int] = None):
        
        super(Vanilla_Model, self).__init__()

        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars
        self.init_range = init_range

        self.s0_nature = s0_nature
        self.train_s0 = train_s0
       

        if n_ext_vars is not None:

            self.n_ext_vars = n_ext_vars

            self.cell = vanilla_cell(n_obs_vars = self.n_obs_vars, 
                                 n_hid_vars= self.n_hid_vars, 
                                 init_range= self.init_range, 
                                 n_ext_vars=n_ext_vars)
            
            self.name = self._generate_model_name(n_ext_vars=n_ext_vars)

        else:

            self.n_ext_vars = None
            self.cell = vanilla_cell(n_obs_vars = self.n_obs_vars, 
                                 n_hid_vars= self.n_hid_vars, 
                                 init_range= self.init_range)
            
            self.name = self._generate_model_name()
        



        # self.cell = vanilla_cell(n_obs_vars = self.n_obs_vars, 
        #                          n_hid_vars= self.n_hid_vars, 
        #                          init_range= self.init_range)
        
        self.device = self.cell._get_default_device()
        # self.name = self._generate_model_name()



        if self.s0_nature.lower() == "zeros_":
            s0 = torch.zeros(1, self.n_state_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            s0 = torch.empty(1, self.n_state_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `s0` a trainable parameter (single vector, not repeated for batch size)
        self.s0 = nn.Parameter(s0, requires_grad=self.train_s0)

    def _generate_model_name(self, n_ext_vars: Optional[int] =None) -> str:
        """Generates a unique model name based on configuration."""

        if n_ext_vars is not None:
            name = f"VanillaModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}_ext{n_ext_vars}"
        else:
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
                ext_data_window: Optional[torch.Tensor]=None ,
                forecast_horizon: Optional[int] = None , 
                future_externals: Optional[torch.Tensor]=None ) -> VanillaHCNNForwardOutput:
        """
        Executes a forward pass through the Vanilla HCNN model over a sequence of observations.
        Supports both teacher-forced training and optional auto-regressive forecasting.

        Parameters
        ----------
        `data_window` : torch.Tensor
            Observed input sequence of shape (`batch_size`, `sequence_length`, `n_obs_vars`).
            Represents a batch of multivariate time series.

        `forecast_horizon` : Optional[int], default=None
            Number of future steps to forecast beyond the observed `data_window`.
            If None, no forecasting is performed. If specified, the model enters
            auto-regressive prediction mode for `forecast_horizon` steps.
        
        `ext_data_window` : torch.Tensor
            External variables sequence of shape (`batch_size`, `sequence_length`, `n_ext_vars`).
            Represents a batch of multivariate time series accounting for the external variables.

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

        if self.n_ext_vars is not None:


            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :],
                        externals=ext_data_window[:, t, :] )
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term

                # Final observed time step

                # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
                # expectations[:, seq_length - 1, :] = last_y_hat
                # if ext_data_window is not None:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                            self.cell.B(ext_data_window[:,seq_length - 1, :]), self.cell.ConMat.T)
            else:

                                            
                    # Use external variables for the last time step
                    # 
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                          self.cell.B(ext_data_window[:,seq_length - 1, :]), self.cell.ConMat.T)
                
            # last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            #     # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
                

            # expectations[:, seq_length - 1, :] = last_y_hat
            # delta_terms[:, seq_length - 1, :] = last_delta_term
            # partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
        else:

            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :] )
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term

                # Final observed time step

                # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
                # expectations[:, seq_length - 1, :] = last_y_hat
                # if ext_data_window is not None:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
            else:

                                            
                    # Use external variables for the last time step
                    # 
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
                
        last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
            

        expectations[:, seq_length - 1, :] = last_y_hat
        delta_terms[:, seq_length - 1, :] = last_delta_term
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

                future_states[:, 0, :] = states[:, seq_length - 1, :]

                if self.n_ext_vars is not None:


                # Use the last observed state as the starting point
                    

                    # Forecast future steps
                    for t in range(1, forecast_horizon):
                        forecast, next_state, _ = self.cell( state=future_states[:, t - 1, :],
                                                            teacher_forcing=False , 
                                                            externals = future_externals[:, t - 1, :] )
                        
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state

                    forecasts[:,t] = torch.matmul(future_states[:, t, :] + self.cell.B(future_externals[:, t - 1, :] ), self.cell.ConMat.T)

                else:

                    # future_states[:, 0, :] = states[:, seq_length - 1, :]

                    # Forecast future steps
                    for t in range(1, forecast_horizon):
                        
                        forecast, next_state, _ = self.cell( state=future_states[:, t - 1, :],
                                                            teacher_forcing=False )
                        
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state

                    forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)
                    

        return VanillaHCNNForwardOutput( expectations=expectations, states=states, delta_terms=delta_terms, forecasts=forecasts, future_states=future_states )


    # def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer, checkpoint_dir: str = "checkpoints"):
    #     """Saves model checkpoint to specified directory if performance improves."""
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_epoch{epoch}.pth")
    #     torch.save({
    #         "epoch": epoch,
    #         "model_state_dict": self.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": loss
    #     }, checkpoint_path)
    #     print(f"✅ Checkpoint saved at {checkpoint_path}")

    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer, checkpoint_dir: str = "checkpoints", cleanup: bool = False):
        """Saves model checkpoint. Optionally removes older ones."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        if cleanup:
            # Remove previous checkpoints for this model
            pattern = os.path.join(checkpoint_dir, f"{self.name}_epoch*.pth")
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old checkpoint: {f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")
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






class PTF_Model(nn.Module):
    """
    Partial Teacher Forcing HCNN Wrapper Model (HCNN-pTF).

    This model wraps the `ptf_cell` to provide a training and forecasting interface 
    that incorporates **partial teacher forcing** via dropout scaling. It supports:
    - Full teacher forcing on observed data
    - Dropout-controlled blending of predicted and observed deltas (`y_hat - y_true`)
    - Optional forecasting beyond the observed horizon using autonomous rollouts

    Parameters
    ----------
    n_obs_vars : int
        Number of observed variables (i.e., dimensionality of model output at each time step).

    n_hid_vars : int
        Number of hidden (latent) variables in the state.

    `n_ext_vars` : int
        Number of external variables (optional). 

    s0_nature : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `s0`:
        - 'zeros_': All-zero initialization.
        - 'random_': Uniform random initialization within `init_range`.

    train_s0 : bool
        Whether the initial hidden state `s0` is a learnable parameter.


    init_range : Tuple[float, float], optional
        Tuple specifying the range for uniform weight initialization.
        Also used when `s0_nature='random_'`. Default is (-0.75, 0.75).

    target_prob : float, optional
        Maximum dropout probability for partial teacher forcing, used to compute delta scaling
        in the later epochs of training. Default is 0.25.

    drop_output : bool, optional
        If True, indicates whether to apply dropout also to the model output.
        (Currently reserved for future use.)

    Direct Attributes (from inputs)
    -------------------------------
    n_obs : int
        Number of observed variables.

    n_hid_vars : int
        Number of hidden variables.
    
    `n_ext_vars` : int
        Number of external variables (optional). 

    s0_nature : str
        Initial hidden state strategy.

    train_s0 : bool
        Whether `s0` is learnable.


    init_range : Tuple[float, float]
        Initialization range for both weights and optionally `s0`.

    target_prob : float
        Maximum dropout probability for teacher forcing adjustment.

    drop_output : bool
        Flag for extending dropout logic to output layer.

    Indirect Attributes (initialized internally)
    --------------------------------------------
    cell : ptf_cell
        The core computation cell that implements dropout-modulated teacher forcing.

    s0 : nn.Parameter
        Initial hidden state of shape (1, n_hid_vars). Shared across batch.
        May be fixed or trainable depending on `train_s0`.

    name : str
        A unique name generated from model configuration for saving and loading.

    device : torch.device
        Computation device used by the model (CUDA, MPS, or CPU).

    Methods
    -------
    forward(data_window: torch.Tensor, forecast_horizon: Optional[int] = None, prob: float = 0.0) -> PTFHCNNForwardOutput
        Executes a forward pass through the model using partial teacher forcing.

    initial_hidden_state() -> nn.Parameter
        Returns the trainable initial hidden state `s0`.

    decrease_dropout_prob(current_epoch: int, num_epochs: int, prob: float) -> float
        Adjusts the dropout probability linearly during training.
        Increases from 0 to `target_prob` over the second half of training epochs.

    save_checkpoint(epoch: int, loss: float, optimizer: torch.optim.Optimizer)
        Saves model weights, optimizer state, and training loss to a checkpoint file.

    Notes
    -----
    - Dropout is applied to the delta term (`y_true - y_hat`) before state update.
    - This allows a gradual relaxation from supervised to autonomous dynamics during training.
    - The same initial hidden state is used across all batch items unless `train_s0` is True.
    """


    def __init__(self, n_obs_vars: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool,
                 init_range: Tuple[float, float] = (-0.75, 0.75), 
                 target_prob: float = 0.25 ,
                 drop_output:bool = False,
                n_ext_vars : Optional[int] = None):
        
        super(PTF_Model, self).__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars

        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        
        self.init_range = init_range
        self.target_prob = target_prob
        self.drop_output =  drop_output

        if n_ext_vars is not None:

            self.n_ext_vars = n_ext_vars


            self.cell = ptf_cell(n_obs_vars = self.n_obs_vars,
                                  n_hid_vars =self.n_hid_vars, 
                                init_range = self.init_range,
                                n_ext_vars=self.n_ext_vars)

            self.name = self._generate_model_name(n_ext_vars=n_ext_vars)
  
        else:
            self.n_ext_vars = None
            self.cell = ptf_cell(n_obs_vars = self.n_obs_vars, 
                                 n_hid_vars =self.n_hid_vars, 
                                init_range = self.init_range)
                            
            self.name = self._generate_model_name()

   
        # self.name = self._generate_model_name()
        self.device = self.cell._get_default_device()


        if self.s0_nature.lower() == "zeros_":
            s0 = torch.zeros(1, self.n_state_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            s0 = torch.empty(1, self.n_state_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `s0` a trainable parameter (single vector, not repeated for batch size)
        self.s0 = nn.Parameter(s0, requires_grad=self.train_s0)

    def _generate_model_name(self , n_ext_vars: Optional[int] =None) -> str:
        """Generates a unique model name based on configuration."""


        if self.n_ext_vars is not None:
            name = f"PTFModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}_ext{n_ext_vars}"
        else:
            name = f"PTFModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
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
    
    def decrease_dropout_prob(self,  current_epoch :int , num_epochs:int, prob:float  )-> float:

        """This function adjusts the dropout probability based on the current epoch
        Input shape:
        current_epoch : int

        the delta term that is added up to the current probability after each epoch 
        as from  num_epochs/2 is calculated as:

        delta = target_prob / (num_epochs / 2)

        
        Output shape:
        prob : float"""
        # self.target_prob = target_prob
        # self.num_epochs = num_epochs

        #delta = self.target_prob / (num_epochs / 2)


        if current_epoch >= (num_epochs / 2):
            new_prob = min(self.target_prob,  prob + (self.target_prob / (num_epochs / 2)))
            return new_prob

        else:
            new_prob = 0.
            return new_prob
        
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
        


    def forward(self, data_window: torch.Tensor,
                prob: float = 0.0,
                ext_data_window: Optional[torch.Tensor]=None ,
                forecast_horizon: Optional[int] = None , 
                future_externals: Optional[torch.Tensor]=None ) -> HCNNpTFForwardOutput:
        """
        Executes a forward pass through the Partial Teacher Forcing HCNN model (HCNN-pTF).
        Supports dynamic control over the amount of teacher forcing via dropout scaling,
        as well as optional future forecasting.

        Parameters
        ----------
        data_window : torch.Tensor
            Input observation sequence of shape (batch_size, sequence_length, n_obs_vars).
            Each sample in the batch is a multivariate time series.

        forecast_horizon : Optional[int], default=None
            If specified, the model forecasts this number of future time steps beyond the
            provided `data_window` using auto-regressive unrolling.

        `ext_data_window` : torch.Tensor
            External variables sequence of shape (`batch_size`, `sequence_length`, `n_ext_vars`).
            Represents a batch of multivariate time series accounting for the external variables.

        prob : float, default=0.0
            Dropout probability used in partial teacher forcing. A higher value corresponds
            to weaker guidance from the observed data (`y_true`) during state transitions.

        Returns
        -------
        PTFHCNNForwardOutput
            A namedtuple containing:

            - expectations : torch.Tensor
                Predicted outputs for the observed sequence.
                Shape: (batch_size, sequence_length, n_obs_vars)

            - states : torch.Tensor
                Hidden state vectors for the observed sequence.
                Shape: (batch_size, sequence_length, n_hid_vars)

            - delta_terms : torch.Tensor
                Differences between ground-truth and predicted observations (y_true - y_hat).
                Shape: (batch_size, sequence_length, n_obs_vars)

            - partial_delta_terms : torch.Tensor
                Delta terms after applying dropout (i.e., scaled teacher forcing error).
                Shape: (batch_size, sequence_length, n_obs_vars)

            - forecasts : Optional[torch.Tensor]
                Predicted outputs for `forecast_horizon` future steps.
                Shape: (batch_size, forecast_horizon, n_obs_vars). Returns None if no forecasting.

            - future_states : Optional[torch.Tensor]
                Hidden state vectors for future steps.
                Shape: (batch_size, forecast_horizon, n_hid_vars). Returns None if no forecasting.

        Notes
        -----
        - Dropout is applied on the delta term (`y_true - y_hat`) before updating the internal state.
        - Forecasting starts from the last state and proceeds without teacher forcing.
        - The dropout is applied independently at each time step when `teacher_forcing` is active.
        """

        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_state_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)
        partial_delta_terms = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.s0



        if self.n_ext_vars is not None:
            


            if seq_length > 1:

            # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term , partial_delta_term= self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True, prob=prob,
                        observation=data_window[:, t, :],
                        externals=ext_data_window[:, t, :])
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term
                    partial_delta_terms[:, t, :] = partial_delta_term

            # Final observed time step


                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                          self.cell.B(ext_data_window[:,seq_length - 1, :]), self.cell.ConMat.T)
            else:


                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                          self.cell.B(ext_data_window[:,seq_length - 1, :]), self.cell.ConMat.T)

            
            # last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
            

            # expectations[:, seq_length - 1, :] = last_y_hat
            # delta_terms[:, seq_length - 1, :] = last_delta_term
            # partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term

        else:

            if seq_length > 1:
                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term , partial_delta_term= self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True, prob=prob,
                        observation=data_window[:, t, :])
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term
                    partial_delta_terms[:, t, :] = partial_delta_term
                                 
                # Use external variables for the last time step
                # 
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
            

            else:

                # Use external variables for the last time step
                # 
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
            # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            
        last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
        last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)

        expectations[:, seq_length - 1, :] = last_y_hat
        delta_terms[:, seq_length - 1, :] = last_delta_term
        partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
            


        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs_vars, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_state_vars, device=self.device)

            teach_forc = torch.matmul(last_partial_delta_term,self.cell.ConMat)
            r_state = states[:, 0, :] - teach_forc
            next_state = self.cell.A(torch.tanh(r_state))

            with torch.no_grad():

                future_states[:, 0, :] = states[:, seq_length - 1, :]    


                if self.n_ext_vars is not None:
                # Use the last observed state as the starting point


                # Use the last observed state as the starting point
                    future_states[:, 0, :] = states[:, seq_length - 1, :]

                    # Forecast future steps
                    for t in range(1, forecast_horizon):

                        forecast, next_state, _ , __= self.cell( state=future_states[:, t - 1, :],
                                                            teacher_forcing=False , 
                                                            externals = future_externals[:, t - 1, :] ,
                                                            prob = .0)
                        

                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state
                        
                    forecasts[:,t] = torch.matmul(future_states[:, t, :] + self.cell.B(future_externals[:, t - 1, :] ),
                                self.cell.ConMat.T)

                else:

                    
                    for t in range(1, forecast_horizon):
                        
                        forecast, next_state, _, __ = self.cell( state=future_states[:, t - 1, :],
                                                            teacher_forcing=False, prob  = 0. )
                        
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state


                    forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)

        return HCNNpTFForwardOutput( expectations=expectations, states=states, delta_terms=delta_terms,
                                    partial_delta_terms =partial_delta_terms, forecasts=forecasts, future_states=future_states )

    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer, checkpoint_dir: str = "checkpoints", cleanup: bool = False):
        """Saves model checkpoint. Optionally removes older ones."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        if cleanup:
            # Remove previous checkpoints for this model
            pattern = os.path.join(checkpoint_dir, f"{self.name}_epoch*.pth")
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old checkpoint: {f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")

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










class LForm_Model(nn.Module):
    """
    LSTM-Formulation HCNN Wrapper Model (HCNN-LForm).

    This model wraps the `lstm_cell`, a nonlinear HCNN cell that introduces memory-preserving 
    behavior inspired by LSTM dynamics. It incorporates a residual-corrected state transition 
    and a learnable diagonal matrix (`D`) for modulating long-term dependencies.

    The model supports full teacher forcing for supervised sequence training and can forecast
    future states in an auto-regressive fashion without additional supervision.

    Parameters
    ----------
    `n_obs_vars` : int
        Number of observable variables (i.e., output dimensionality).

    `n_hid_vars` : int
        Number of hidden (latent) variables in the state.

    `n_ext_vars` : int
        Number of external variables (optional). 


    `s0_nature` : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `s0`.
        - 'zeros_': Initialize as a zero vector.
        - 'random_': Initialize uniformly using `init_range`.

    `train_s0` : bool
        If True, the initial state `s0` is a trainable parameter.


    `init_range` : Tuple[float, float], optional
        Range for uniform initialization when `s0_nature='random_'`. Also passed to the internal cell.

        
    Direct Attributes (from inputs)
    -------------------------------
    `n_obs_vars` : int
        Number of observed variables.

    `n_hid_vars` : int
        Number of hidden variables.

    `n_ext_vars` : int
        Number of external variables (optional). 

    `s0_nature` : str
        Initial state setup strategy ('zeros_' or 'random_').

    `train_s0` : bool
        Whether the initial hidden state `s0` is trainable.



    `init_range` : Tuple[float, float]
        Initialization range for random initialization.


    Indirect Attributes (initialized internally)
    --------------------------------------------

    `cell` : lstm_cell
        The core recurrent computation unit of the model..

    s0 : torch.nn.Parameter
        The initial hidden state tensor of shape (1, n_hid_vars).
        Repeated across the batch in the forward pass.

    `name` : str
        A unique model name based on initialization and training configuration.

    `device` : torch.device
        Computation device (CUDA, MPS, or CPU) used by the model.


    Methods
    -------
    forward(data_window: torch.Tensor, forecast_horizon: Optional[int] = None) -> HCNNLFormForwardOutput
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
                 train_s0: bool,
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 init_diag: float = 1.0,
                n_ext_vars : Optional[int] = None):
        
        super(LForm_Model, self).__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars
        self.n_state_vars = self.n_hid_vars + self.n_obs_vars

        self.s0_nature = s0_nature
        self.train_s0 = train_s0

        self.init_range = init_range
        self.init_diag = init_diag

        if n_ext_vars is not None:

            self.n_ext_vars = n_ext_vars

            self.cell = lstm_cell(n_obs_vars= self.n_obs_vars,n_hid_vars= self.n_hid_vars,
                               init_range = self.init_range, init_diag=self.init_diag,
                               n_ext_vars = n_ext_vars)
            
            self.name = self._generate_model_name(n_ext_vars=n_ext_vars)

            
        else:
            self.n_ext_vars = None
            self.cell = lstm_cell(n_obs_vars= self.n_obs_vars,n_hid_vars= self.n_hid_vars,
                               init_range = self.init_range, init_diag=self.init_diag
                               )
            self.name = self._generate_model_name()
        self.device = self.cell._get_default_device()



        if self.s0_nature.lower() == "zeros_":
            s0 = torch.zeros(1, self.n_state_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            s0 = torch.empty(1, self.n_state_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `s0` a trainable parameter (single vector, not repeated for batch size)
        self.s0 = nn.Parameter(s0, requires_grad=self.train_s0)

    def _generate_model_name(self , n_ext_vars: Optional[int] =None) -> str:
        """Generates a unique model name based on configuration."""

        if self.n_ext_vars is not None:
            name = f"LFormModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}_ext{n_ext_vars}"
        
        else:

            name = f"LFormModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
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
                ext_data_window: Optional[torch.Tensor]=None ,
                forecast_horizon: Optional[int] = None,
                future_externals: Optional[torch.Tensor]=None ) -> HCNNLFormForwardOutput:
        """
        Executes a forward pass through the HCNNLForm model over a sequence of observations.
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

            
        `ext_data_window` : torch.Tensor
            External variables sequence of shape (`batch_size`, `sequence_length`, `n_ext_vars`).
            Represents a batch of multivariate time series accounting for the external variables.
   
        Returns
        -------
        HCNNLFormForwardOutput
            A namedtuple containing the following tensors:

            - expectations : torch.Tensor
                Predicted outputs for the observed sequence.
                Shape: (batch_size, sequence_length, n_obs)

            - states : torch.Tensor
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
        - The internal `lstm_cell` modulates memory with a diagonal transformation.
        - The final state from the observed sequence is used to initialize future rollouts.
        - Teacher forcing is applied for all observed inputs.
        """

        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_state_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.s0

        if self.n_ext_vars is not None:

            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :],
                        externals=ext_data_window[:, t, :] )
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term

                # Final observed time step
                # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
                # expectations[:, seq_length - 1, :] = last_y_hat

                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                        self.cell.B(ext_data_window[:,seq_length - 1, :]), 
                                        self.cell.ConMat.T)
                

        
            else:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            
            # last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
                # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
                
            # expectations[:, seq_length - 1, :] = last_y_hat
            # delta_terms[:, seq_length - 1, :] = last_delta_term
            # partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
        else:


            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :] )
                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term

                # Final observed time step
                # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
                # expectations[:, seq_length - 1, :] = last_y_hat

                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , 
                                        self.cell.ConMat.T)
                

        
            else:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            



            
        last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
            

        expectations[:, seq_length - 1, :] = last_y_hat
        delta_terms[:, seq_length - 1, :] = last_delta_term

        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs_vars, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_state_vars, device=self.device)

            teach_forc = torch.matmul(last_delta_term,self.cell.ConMat)
            r_state = states[:, seq_length - 1, :] - teach_forc
            lstm_block = self.cell.A(torch.tanh(r_state)) - r_state
            next_state = r_state + self.cell.D(lstm_block)

            with torch.no_grad():

                future_states[:, 0, :] = next_state# states[:, seq_length - 1, :]
                
                if self.n_ext_vars is not None:
                    # Use the last observed state as the starting point
                    # future_states[:, 0, :] = states[:, seq_length - 1, :]

                    # Forecast future steps
                    for t in range(1, forecast_horizon):
                        forecast, next_state, _ = self.cell(
                            state=future_states[:, t - 1, :],
                            teacher_forcing=False,
                            externals=future_externals[:, t - 1, :] )

                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state
                # Use the last observed state as the starting point
            
            
                    forecasts[:,t] = torch.matmul(future_states[:, t, :] + self.cell.B(future_externals[:, t - 1, :] ), self.cell.ConMat.T)

                else:

                # Forecast future steps
                    for t in range(1, forecast_horizon):
                        forecast, next_state, _ = self.cell(
                            state=future_states[:, t - 1, :],
                            teacher_forcing=False
                        )
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state

                    forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)
        return HCNNLFormForwardOutput(expectations=expectations , states=states,delta_terms=delta_terms,
                                      forecasts=forecasts, future_states=future_states)
        # return expectations, states, delta_terms, forecasts, future_states

    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer, checkpoint_dir: str = "checkpoints", cleanup: bool = False):
        """Saves model checkpoint. Optionally removes older ones."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        if cleanup:
            # Remove previous checkpoints for this model
            pattern = os.path.join(checkpoint_dir, f"{self.name}_epoch*.pth")
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old checkpoint: {f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")


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







class LSpa_Model(nn.Module):
    """
    Large Sparse HCNN Wrapper Model (LSpa_Model).

    This model wraps the `LargeSparse_cell`, which supports structured sparsity in the 
    state transition dynamics via masking. 

    The model supports teacher-forced training and optional auto-regressive forecasting.

    Parameters
    ----------
    `n_obs_vars` : int
        Number of observable variables (i.e., output dimensionality).

    `n_hid_vars` : int
        Number of hidden (latent) variables in the state.

    `n_ext_vars` : int
        Number of external variables (optional). 

    `s0_nature` : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `s0`.
        - 'zeros_': Initialize as a zero vector.
        - 'random_': Initialize uniformly using `init_range`.

    `train_s0` : bool
        If True, the initial state `s0` is a trainable parameter.


    `init_range` : Tuple[float, float], optional
        Range for uniform initialization when `s0_nature='random_'`. Also passed to the internal cell.


    mask_type : Literal['non_obs_block', 'random_block']
        Type of sparsity pattern to apply in the weight matrix:
        - 'random_block': Applies uniform sparsity.
        - 'non_obs_block': Sparsity is applied only to non-observable state subspace.

    sparsity_ratio : float, optional
        Fraction of weights to be set to zero in the masked region. Default is 0.25.


    bias : bool, optional
        Whether to include a bias term in the sparse linear transformation. Default is False.

    Direct Attributes (from inputs)
    -------------------------------
    `n_obs_vars` : int
        Number of observed variables.

    `n_hid_vars` : int
        Number of hidden variables.

    `n_ext_vars` : int
        Number of external variables (optional). 

    `s0_nature` : str
        Initial state setup strategy ('zeros_' or 'random_').

    `train_s0` : bool
        Whether the initial hidden state `s0` is trainable.


    `init_range` : Tuple[float, float]
        Initialization range for random initialization.

    `sparsity_ratio` : float
        Desired sparsity level to apply to the weights.

    `mask_type` : str
        Indicates the type of sparsity mask (`random_block` or `non_obs_block`).

    bias : bool
        Whether sparse transformation includes a bias term.

    Indirect Attributes (initialized internally)
    --------------------------------------------
    cell : LargeSparse_cell
        Core sparse recurrent cell with configurable sparsity.

    s0 : nn.Parameter
        Initial hidden state (shape: [1, n_hid_vars]), learnable if `train_s0=True`.

    device : torch.device
        Computation device used by the model.

    name : str
        Unique model identifier constructed from configuration.

    Methods
    -------
    forward(data_window: torch.Tensor, forecast_horizon: Optional[int] = None) -> HCNNLSPaForwardOutput
        Executes training and optional forecasting steps.

    initial_hidden_state() -> nn.Parameter
        Returns the initial hidden state tensor `s0`.

    save_checkpoint(epoch: int, loss: float, optimizer: torch.optim.Optimizer)
        Saves the model and optimizer state to a checkpoint.

    load_checkpoint(checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None)
        Loads model (and optionally optimizer) state from a checkpoint.
    """


    def __init__(self, n_obs_vars: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, 
                 mask_type: str = Literal['non_obs_block', 'random_block'] , 
                 sparsity_ratio : float = 0.25 ,
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 bias :bool =False,
                n_ext_vars : Optional[int] = None):
        
        super(LSpa_Model, self).__init__()
        self.n_obs_vars = n_obs_vars
        self.n_hid_vars = n_hid_vars

        self.s0_nature = s0_nature
        self.train_s0 = train_s0

        self.init_range = init_range
        self.mask_type = mask_type

        self.sparsity_ratio =  sparsity_ratio

        self.bias = bias


        if n_ext_vars is not None:

            self.n_ext_vars = n_ext_vars

            self.cell = LargeSparse_cell(n_obs_vars =self.n_obs_vars, 
                                    n_hid_vars = self.n_hid_vars,
                                    bias = self.bias,
                                init_range = self.init_range, 
                                sparsity_ratio=self.sparsity_ratio, 
                                mask_type = mask_type,
                                n_ext_vars = self.n_ext_vars)
            
            self.name = self._generate_model_name(n_ext_vars=n_ext_vars)
            
        else:
            
            self.n_ext_vars = None
            self.cell = LargeSparse_cell(n_obs_vars =self.n_obs_vars, 
                                        n_hid_vars = self.n_hid_vars,
                                        bias = self.bias,
                                        init_range = self.init_range, 
                                        sparsity_ratio=self.sparsity_ratio, 
                                        mask_type = mask_type)
            self.name = self._generate_model_name()
        self.device = self.cell._get_default_device()





        if self.s0_nature.lower() == "zeros_":
            s0 = torch.zeros(1, self.cell.n_state_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            s0 = torch.empty(1, self.cell.n_state_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `s0` a trainable parameter (single vector, not repeated for batch size)
        self.s0 = nn.Parameter(s0, requires_grad=self.train_s0)


    def _generate_model_name(self , n_ext_vars: Optional[int] =None)  -> str:
        """Generates a unique model name based on configuration."""

        if self.n_ext_vars is not None:
            name = f"LSpaModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}_ext{n_ext_vars}"
        
        else:

            name = f"LSpaModel_obs{self.n_obs_vars}_hid{self.n_hid_vars}"
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
                ext_data_window: Optional[torch.Tensor]=None ,
                forecast_horizon: Optional[int] = None ,
                future_externals: Optional[torch.Tensor]=None ) -> HCNNLSpaForwardOutput:
        """
        Executes a forward pass through the HCNNLSpa model over a sequence of observations.
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

        `ext_data_window` : torch.Tensor
            External variables sequence of shape (`batch_size`, `sequence_length`, `n_ext_vars`).
            Represents a batch of multivariate time series accounting for the external variables.

        Returns
        -------
        HCNNLSpaForwardOutput
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
        - The last hidden state of the input sequence seeds the forecasting phase.
        - Forecasting uses no teacher forcing and relies purely on autoregressive rollout.
        - The core `LargeSparse_cell` enforces a structured sparse transition.
        """

        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.cell.n_state_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs_vars, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.s0

        if self.n_ext_vars is not None:


            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :],
                        externals=ext_data_window[:, t, :] )

                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term

            # Final observed time step
            # last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
            # expectations[:, seq_length - 1, :] = last_y_hat

            # if ext_data_window is not None:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] + 
                                          self.cell.B(ext_data_window[:,seq_length - 1, :]), self.cell.ConMat.T)
            else:

                                        
                # Use external variables for the last time step
                # 
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
            
            # last_delta_term = data_window[:, seq_length - 1, :] - last_y_hat
            # # last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)
            

            # expectations[:, seq_length - 1, :] = last_y_hat
            # delta_terms[:, seq_length - 1, :] = last_delta_term
            # partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
        else:

            if seq_length > 1:

                # Process observed data
                for t in range(seq_length - 1):
                    expectation, next_state, delta_term = self.cell(
                        state=states[:, t, :],
                        teacher_forcing=True,
                        observation=data_window[:, t, :] )

                    
                    expectations[:, t, :] = expectation
                    states[:, t + 1, :] = next_state
                    delta_terms[:, t, :] = delta_term
        
            # if ext_data_window is not None:
                last_y_hat = torch.matmul(states[:, seq_length - 1, :] , self.cell.ConMat.T)
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
            future_states = torch.zeros(batch_size, forecast_horizon, self.cell.n_state_vars, device=self.device)
            teach_forc = torch.matmul(last_delta_term,self.cell.ConMat)
            r_state = states[:, 0, :] - teach_forc
            next_state = self.cell.Sparse_A(torch.tanh(r_state))
            with torch.no_grad():

                # if ext_data_window is not None:
                # Use the last observed state as the starting point
                future_states[:, 0, :] = next_state#states[:, seq_length - 1, :]

                if self.n_ext_vars is not None:
                # Forecast future steps
                    for t in range(1, forecast_horizon):
                        forecast, next_state, _ = self.cell(
                            state=future_states[:, t - 1, :],
                            teacher_forcing=False,
                            externals = future_externals[:, t - 1, :] )
                    
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state

                    forecasts[:,t] = torch.matmul(future_states[:, t, :] + self.cell.B(future_externals[:, t - 1, :] ), self.cell.ConMat.T)
            
                else:


                    # future_states[:, 0, :] = states[:, seq_length - 1, :]    

                    
                    for t in range(1, forecast_horizon):
                        
                        forecast, next_state, _ = self.cell(
                            state=future_states[:, t - 1, :],
                            teacher_forcing=False
                        )
                        forecasts[:, t - 1, :] = forecast
                        future_states[:, t, :] = next_state

                    forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)
        # return expectations, states, delta_terms, forecasts, future_states 
        return HCNNLSpaForwardOutput( expectations=expectations, states=states,
                                     delta_terms=delta_terms, forecasts=forecasts,
                                     future_states=future_states)


    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer, checkpoint_dir: str = "checkpoints", cleanup: bool = False):
        """Saves model checkpoint. Optionally removes older ones."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        if cleanup:
            # Remove previous checkpoints for this model
            pattern = os.path.join(checkpoint_dir, f"{self.name}_epoch*.pth")
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old checkpoint: {f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")



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

