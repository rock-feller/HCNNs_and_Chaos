import os
import torch
from torch import nn
from typing import Literal, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

class RNNBaseModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 s0_nature: Literal['zeros_', 'random_'], train_s0: bool, 
                 init_range: Tuple[float, float] = (-0.75, 0.75), num_layers: int = 1): #batch_size: int = 1,
        super(RNNBaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        # self.batch_size = batch_size
        self.init_range_s0 = init_range
        self.num_layers = num_layers

        # RNN layer
        self.state_transition_layer = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, bias=False)
        
        # Output layer
        self.output_extraction_layer = nn.Linear(hidden_size, output_size, bias=False)

        # Device handling
        self.device = self._get_default_device()
        self.name = self._generate_model_name()

        # Initialize hidden state
        self.h0 = self.initial_hidden_state()

    def _get_default_device(self) -> torch.device:
        """Determine the default device (CPU, CUDA, or MPS)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _generate_model_name(self) -> str:
        """Generate a unique model name based on configuration."""
        name = f"RNNModel_in{self.input_size}_hid{self.hidden_size}_out{self.output_size}"
        if self.s0_nature == "random_":
            name += f"_randInit{self.init_range_s0[0]}to{self.init_range_s0[1]}"
        else:
            name += "_zeroInit"
        if self.train_s0:
            name += "_trainableS0"
        return name

    def initial_hidden_state(self, batch_size:int = 1) -> nn.Parameter:
        """Initialize the hidden state."""
        if self.s0_nature.lower() == "zeros_":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range_s0
            h0 = torch.empty(self.num_layers, batch_size, self.hidden_size, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")
        
        return nn.Parameter(h0, requires_grad=self.train_s0)
    
    def forward(self, input_sequence: torch.Tensor, initial_states_s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Inputs shape:
          input_sequence: [batch, seq_len, features]
          initial_states_s0: [num_layers, batch, hidden_size]



        The state_transition equation computes the hidden states of the RNN
        across the whole batch of sequences and stack them the following way
        [S_(1), ...., S_(t-2), S_(t-1), S_(t)] where S_(t) is the hidden state at time t
        Then, the outpute equation extracts the output from the hidden states
        and return the outputs as  a batch of [y_(0), ...., y_(t-2), y_(t-1), y_(t)]

        Outputs shape:
        sequence_of_states: [batch, seq_len, hidden_size]
        output_sequence: [batch, seq_len, features]
        hstate_at_T: [num_layers, batch, hidden_size]
        """
        #h0_expanded  = initial_states_s0.unsqueeze(1).expand(-1, batch_size, -1) 
        sequence_of_states, hstate_at_T = self.state_transition_layer(input_sequence, initial_states_s0)
        output_sequence = self.output_extraction_layer(sequence_of_states)
        return sequence_of_states, output_sequence, hstate_at_T
    
    
    def forecast_seq_to_seq(self, input_sequence: torch.Tensor, n_steps: int) -> torch.Tensor:

        """Inputs shape: 
        input_sequence: [seq_len, features]
        initial_states_s0 :  [num_layers, batch_size=1, hidden_size]
        n_steps: len_fcast_horizon
        
        This function takes the input_sequence, compute the forward pass 
        until the present time step for both S_(t) and y_(t) and then
        forecast the future outputs y_(t+1), ...., y_(t+n_steps)
        
        Outputs shape:
        future_outputs: [len_fcast_horizon, features]
            """
        #with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
        initial_states_s0 =  self.initial_hidden_state(batch_size=input_sequence.size(0))
        state_seq, output_seq, hstate_at_T = self.forward(input_sequence, initial_states_s0)

        future_hstates = torch.empty(1, n_steps + 1, initial_states_s0.size(2)).to(self.device)
        future_outputs = torch.empty(1, n_steps + 1, input_sequence.size(2)).to(self.device)

        future_hstates[:, 0, :] = hstate_at_T.clone()
        future_outputs[:, 0, :] = output_seq[:, -1, :]

        for t_step in range(n_steps):
            next_input = future_outputs[:, t_step, :].unsqueeze(1)
            _, output_step, h_future_T = self.forward(next_input, future_hstates[:, t_step, :].unsqueeze(0))
            future_hstates[:, t_step + 1, :] = h_future_T.squeeze(0)
            future_outputs[:, t_step + 1, :] = output_step.squeeze(1)

        return output_seq , future_outputs[0, 1:]



    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
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