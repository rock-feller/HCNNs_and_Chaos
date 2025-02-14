import os
import torch
from torch import nn
from typing import Literal, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from .modules import vanilla_cell , ptf_cell, lstm_cell, LargeSparse_cell


class Vanilla_Model(nn.Module):

    def __init__(self, n_obs: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, batch_size: int = 1,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        
        super(Vanilla_Model, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.batch_size = batch_size
        self.init_range = init_range

        self.cell = vanilla_cell(n_obs, n_hid_vars, init_range)
        self.device = self.cell._get_default_device()
        self.name = self._generate_model_name()



        if self.s0_nature.lower() == "zeros_":
            h0 = torch.zeros(1, n_hid_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            h0 = torch.empty(1, n_hid_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `h0` a trainable parameter (single vector, not repeated for batch size)
        self.h0 = nn.Parameter(h0, requires_grad=self.train_s0)

    def _generate_model_name(self) -> str:
        """Generates a unique model name based on configuration."""
        name = f"VanillaModel_obs{self.n_obs}_hid{self.n_hid_vars}"
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
        return self.h0
    

    def forward(self, data_window: torch.Tensor,
            forecast_horizon: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        """
        Perform a forward pass over a sequence of inputs.

        Parameters:
        -----------
        data_window : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, n_obs).
        forecast_horizon : Optional[int], default=None
            If provided, the model rolls forward for the given number of future time frames.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            - expectations : torch.Tensor
                Predicted outputs of shape (batch_size, sequence_length, n_obs).
            - states : torch.Tensor
                Hidden states of shape (batch_size, sequence_length, n_hid_vars).
            - delta_terms : torch.Tensor
                Delta terms (y_true - y_hat) of shape (batch_size, sequence_length, n_obs).
            - forecasts : Optional[torch.Tensor]
                Forecasted outputs of shape (batch_size, forecast_horizon, n_obs). None if no forecast_horizon.
            - future_states : Optional[torch.Tensor]
                Future hidden states of shape (batch_size, forecast_horizon, n_hid_vars). None if no forecast_horizon.
        """
        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_hid_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.h0

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
        last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
        expectations[:, seq_length - 1, :] = last_y_hat

        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_hid_vars, device=self.device)

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

        return expectations, states, delta_terms, forecasts, future_states

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

class PTF_Model(nn.Module):

    def __init__(self, n_obs: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, batch_size: int = 1,
                 init_range: Tuple[float, float] = (-0.75, 0.75), 
                 target_prob: float = 0.25 ,
                 drop_output:bool = False):
        
        super(PTF_Model, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.batch_size = batch_size
        self.init_range = init_range
        self.target_prob = target_prob
        self.drop_output =  drop_output


        self.cell = ptf_cell(n_obs, n_hid_vars, init_range)
        self.device = self.cell._get_default_device()
        self.name = self._generate_model_name()



        if self.s0_nature.lower() == "zeros_":
            h0 = torch.zeros(1, n_hid_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            h0 = torch.empty(1, n_hid_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `h0` a trainable parameter (single vector, not repeated for batch size)
        self.h0 = nn.Parameter(h0, requires_grad=self.train_s0)

    def _generate_model_name(self) -> str:
        """Generates a unique model name based on configuration."""
        name = f"PTFModel_obs{self.n_obs}_hid{self.n_hid_vars}"
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
        return self.h0
    
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
        


    def forward(self, data_window: torch.Tensor, prob:float,
            forecast_horizon: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        """
        Perform a forward pass over a sequence of inputs.

        Parameters:
        -----------
        data_window : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, n_obs).
        forecast_horizon : Optional[int], default=None
            If provided, the model rolls forward for the given number of future time frames.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            - expectations : torch.Tensor
                Predicted outputs of shape (batch_size, sequence_length, n_obs).
            - states : torch.Tensor
                Hidden states of shape (batch_size, sequence_length, n_hid_vars).
            - delta_terms : torch.Tensor
                Delta terms (y_true - y_hat) of shape (batch_size, sequence_length, n_obs).
            - forecasts : Optional[torch.Tensor]
                Forecasted outputs of shape (batch_size, forecast_horizon, n_obs). None if no forecast_horizon.
            - future_states : Optional[torch.Tensor]
                Future hidden states of shape (batch_size, forecast_horizon, n_hid_vars). None if no forecast_horizon.
        """
        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_hid_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)
        partial_delta_terms = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.h0

        # Process observed data
        for t in range(seq_length - 1):
            expectation, next_state, delta_term , partial_delta_term= self.cell(
                state=states[:, t, :],
                teacher_forcing=True, prob=prob,
                observation=data_window[:, t, :]
            )
            expectations[:, t, :] = expectation
            states[:, t + 1, :] = next_state
            delta_terms[:, t, :] = delta_term
            partial_delta_terms[:, t, :] = partial_delta_term

        # Final observed time step
        last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
        last_delta_term = data_window[:, t, :] - last_y_hat
        last_partial_delta_term = self.cell.ptf_dropout(prob)(last_delta_term)

        expectations[:, seq_length - 1, :] = last_y_hat
        delta_terms[:, seq_length - 1, :] = last_delta_term
        partial_delta_terms[:, seq_length - 1, :]=last_partial_delta_term
            
        
        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_hid_vars, device=self.device)

            with torch.no_grad():
                # Use the last observed state as the starting point
                future_states[:, 0, :] = states[:, seq_length - 1, :]

                # Forecast future steps
                for t in range(1, forecast_horizon):
                    forecast, next_state, _,__ = self.cell(
                        state=future_states[:, t - 1, :],prob=0.,
                        teacher_forcing=False
                    )
                    forecasts[:, t - 1, :] = forecast
                    future_states[:, t, :] = next_state

            forecasts[:,t] = torch.matmul(future_states[:, t, :], self.cell.ConMat.T)

        return expectations, states, delta_terms, partial_delta_terms, forecasts, future_states
    
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

class LForm_Model(nn.Module):

    def __init__(self, n_obs: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, batch_size: int = 1,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        
        super(LForm_Model, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.batch_size = batch_size
        self.init_range = init_range

        self.cell = lstm_cell(n_obs, n_hid_vars, init_range)
        self.device = self.cell._get_default_device()
        self.name = self._generate_model_name()


        if self.s0_nature.lower() == "zeros_":
            h0 = torch.zeros(1, n_hid_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            h0 = torch.empty(1, n_hid_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `h0` a trainable parameter (single vector, not repeated for batch size)
        self.h0 = nn.Parameter(h0, requires_grad=self.train_s0)

    def _generate_model_name(self) -> str:
        """Generates a unique model name based on configuration."""
        name = f"LFormModel_obs{self.n_obs}_hid{self.n_hid_vars}"
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
        return self.h0
    

    def forward(self, data_window: torch.Tensor,
            forecast_horizon: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        """
        Perform a forward pass over a sequence of inputs.

        Parameters:
        -----------
        data_window : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, n_obs).
        forecast_horizon : Optional[int], default=None
            If provided, the model rolls forward for the given number of future time frames.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            - expectations : torch.Tensor
                Predicted outputs of shape (batch_size, sequence_length, n_obs).
            - states : torch.Tensor
                Hidden states of shape (batch_size, sequence_length, n_hid_vars).
            - delta_terms : torch.Tensor
                Delta terms (y_true - y_hat) of shape (batch_size, sequence_length, n_obs).
            - forecasts : Optional[torch.Tensor]
                Forecasted outputs of shape (batch_size, forecast_horizon, n_obs). None if no forecast_horizon.
            - future_states : Optional[torch.Tensor]
                Future hidden states of shape (batch_size, forecast_horizon, n_hid_vars). None if no forecast_horizon.
        """
        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_hid_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.h0

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
        last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
        expectations[:, seq_length - 1, :] = last_y_hat

        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_hid_vars, device=self.device)

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

        return expectations, states, delta_terms, forecasts, future_states

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

class LSpa_Model(nn.Module):

    def __init__(self, n_obs: int, n_hid_vars: int,
                 s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, batch_size: int = 1,
                 mask_type: str = "random" , 
                 sparsity : float = 0.25 ,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        
        super(LSpa_Model, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.batch_size = batch_size
        self.init_range = init_range
        self.mask_type = mask_type
        self.sparsity =  sparsity

        self.cell = LargeSparse_cell(n_obs, n_hid_vars, init_range, sparsity=sparsity, mask_type = mask_type)

        self.device = self.cell._get_default_device()

        self.name = self._generate_model_name()



        if self.s0_nature.lower() == "zeros_":
            h0 = torch.zeros(1, n_hid_vars, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            h0 = torch.empty(1, n_hid_vars, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        # Make `h0` a trainable parameter (single vector, not repeated for batch size)
        self.h0 = nn.Parameter(h0, requires_grad=self.train_s0)


    def _generate_model_name(self) -> str:
        """Generates a unique model name based on configuration."""
        name = f"LSpaModel_obs{self.n_obs}_hid{self.n_hid_vars}"
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
        return self.h0
    

    def forward(self, data_window: torch.Tensor,
            forecast_horizon: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        """
        Perform a forward pass over a sequence of inputs.

        Parameters:
        -----------
        data_window : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, n_obs).
        forecast_horizon : Optional[int], default=None
            If provided, the model rolls forward for the given number of future time frames.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
            - expectations : torch.Tensor
                Predicted outputs of shape (batch_size, sequence_length, n_obs).
            - states : torch.Tensor
                Hidden states of shape (batch_size, sequence_length, n_hid_vars).
            - delta_terms : torch.Tensor
                Delta terms (y_true - y_hat) of shape (batch_size, sequence_length, n_obs).
            - forecasts : Optional[torch.Tensor]
                Forecasted outputs of shape (batch_size, forecast_horizon, n_obs). None if no forecast_horizon.
            - future_states : Optional[torch.Tensor]
                Future hidden states of shape (batch_size, forecast_horizon, n_hid_vars). None if no forecast_horizon.
        """
        batch_size, seq_length, _ = data_window.size()

        # Initialize tensors for observed data
        states = torch.zeros(batch_size, seq_length, self.n_hid_vars, device=self.device)
        expectations = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)
        delta_terms = torch.zeros(batch_size, seq_length, self.n_obs, device=self.device)

        # Use the same initial hidden state for all sequences in the batch
        states[:, 0, :] = self.h0

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
        last_y_hat = torch.matmul(states[:, seq_length - 1, :], self.cell.ConMat.T)
        expectations[:, seq_length - 1, :] = last_y_hat

        # Initialize tensors for forecasts
        forecasts = None
        future_states = None

        if forecast_horizon:
            forecasts = torch.zeros(batch_size, forecast_horizon, self.n_obs, device=self.device)
            future_states = torch.zeros(batch_size, forecast_horizon, self.n_hid_vars, device=self.device)

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

        return expectations, states, delta_terms, forecasts, future_states 

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
