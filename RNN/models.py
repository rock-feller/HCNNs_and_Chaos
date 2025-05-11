import os, glob
import torch
from torch import nn
from typing import Literal, Tuple, Optional
from collections import namedtuple

# Named tuple for the output of the RNN model
RNNForwardOutput = namedtuple(
    "RNNForwardOutput",
    ["hidden_states", "output_sequence", "final_hidden_state", "forecasts"]
)

class RNN_Model(nn.Module):
    """
    Wrapper Model for a basic RNN sequence-to-sequence forecaster.

    This model wraps a PyTorch `nn.RNN` layer with configurable initial hidden state (`h0`)
    and provides interfaces for both full-sequence forward passes and auto-regressive forecasting.

    Parameters
    ----------
    `input_size` : int
        Dimensionality of the input features (number of observable variables).

    `hidden_size` : int
        Number of hidden units in the RNN.

    `output_size` : int
        Dimensionality of the output space (usually equal to input_size).

    `s0_nature` : Literal['zeros_', 'random_']
        Strategy for initializing the initial hidden state `h0`.
        - 'zeros_': Initialize as a zero tensor.
        - 'random_': Initialize uniformly in `init_range`.

    `train_s0` : bool
        Whether the initial hidden state `h0` is a trainable parameter.

    `init_range` : Tuple[float, float], optional
        Range for uniform initialization when `s0_nature='random_'`.

    `num_layers` : int
        Number of stacked RNN layers.

    Attributes
    ----------
    `state_transition_layer` : nn.RNN
        Core recurrent computation unit.

    `output_extraction_layer` : nn.Linear
        Maps hidden states to output space.

    `device` : torch.device
        The device used (CPU, CUDA, or MPS).

    `name` : str
        A unique model name based on configuration.

    `h0` : torch.nn.Parameter
        Initial hidden state vector, broadcast across batch during forward pass.

    Methods
    -------
    forward(input_sequence: torch.Tensor, initial_states_s0: torch.Tensor) -> RNNForwardOutput
        Executes full forward pass and returns hidden states, outputs, and final hidden state.

    forecast_seq_to_seq(input_sequence: torch.Tensor, n_steps: int) -> torch.Tensor
        Auto-regressively forecasts `n_steps` future outputs starting from final output.

    initial_hidden_state(batch_size: int = 1) -> nn.Parameter
        Returns a properly initialized (and optionally trainable) hidden state tensor.

    save_checkpoint(epoch: int, loss: float, optimizer: torch.optim.Optimizer) -> None
        Saves model and optimizer state.

    load_checkpoint(checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]
        Loads model and optimizer state from checkpoint.
    """

    def __init__(self, input_size: int, hidden_size: int,   output_size: int,
                 s0_nature: Literal['zeros_', 'random_'], train_s0: bool,
                 init_range: Tuple[float, float] = (-0.75, 0.75), num_layers: int = 1,  ext_vars: Optional[int] =None):
        
        super(RNN_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.init_range = init_range
        self.num_layers = num_layers

        if ext_vars is not None:
            self.ext_vars = ext_vars
            self.input_size = input_size + ext_vars
        else:
            self.ext_vars =  None
            self.input_size = input_size

        self.state_transition_layer = nn.RNN(self.input_size, hidden_size, num_layers=num_layers, batch_first=True, bias=False)
        self.output_extraction_layer = nn.Linear(hidden_size, output_size, bias=False)

        self.device = self._get_default_device()
        self.name = self._generate_model_name()

        # In __init__, after setting device and name
        if self.s0_nature.lower() == "zeros_":
            h0_tensor = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        elif self.s0_nature.lower() == "random_":
            low, high = self.init_range
            h0_tensor = torch.empty(self.num_layers, 1, self.hidden_size, device=self.device).uniform_(low, high)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

        self.initial_hidden_state = nn.Parameter(h0_tensor, requires_grad=self.train_s0)


        # self.h0 = self.initial_hidden_state()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _generate_model_name(self) -> str:
        name = f"RNNModel_in{self.input_size}_hid{self.hidden_size}_out{self.output_size}"
        if self.s0_nature == "random_":
            name += f"_randInit{self.init_range[0]}to{self.init_range[1]}"
        else:
            name += "_zeroInit"
        if self.train_s0:
            name += "_trainableS0"
        return name


    def forward(self, input_sequence: torch.Tensor,
                initial_states: Optional[torch.Tensor] = None,
                batch_of_externals: Optional[torch.Tensor] = None) -> RNNForwardOutput:
        """
        Executes a full-sequence forward pass through the RNN model.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Shape: (batch, seq_len, input_size)

        initial_states_s0 : Optional[torch.Tensor]
            Initial hidden state, shape: (num_layers, batch, hidden_size).
            If None, uses internal learned `self.initial_hidden_state`.

        batch_of_externals : Optional[torch.Tensor]
            External variables to concatenate to inputs.
            Shape: (batch, seq_len, ext_dim)

        Returns
        -------
        RNNForwardOutput
            Named tuple with:
            - hidden_states: (batch, seq_len, hidden_size)
            - output_sequence: (batch, seq_len, output_size)
            - final_hidden_state: (num_layers, batch, hidden_size)
            - forecasts: None
        """
        # batch_size = input_sequence.size(0)

        if batch_of_externals is not None:
            input_sequence = torch.cat([input_sequence, batch_of_externals], dim=2)


        

        hidden_states, h_T = self.state_transition_layer(input_sequence, initial_states)
        output_sequence = self.output_extraction_layer(hidden_states)

        return RNNForwardOutput( hidden_states=hidden_states, output_sequence=output_sequence,
                                final_hidden_state=h_T, forecasts=None)


    def forecast_seq_to_seq(self, calibration_window: torch.Tensor, n_steps: int,
                            externals_for_calibration: Optional[torch.Tensor] = None ,
                            externals_for_forecasts: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
        Forecasts future steps using an auto-regressive manner.

        Parameters
        ----------
        calibration_window : torch.Tensor
            Shape: (seq_len, input_size)

        n_steps : int
            Number of future time steps to forecast.

        externals_for_calibration : Optional[torch.Tensor]
            External variables of shape (seq_len, ext_vars)

        externals_for_forecasts : Optional[torch.Tensor]
            External variables of shape (n_steps, ext_vars)


        Returns
        -------
        torch.Tensor
            Forecasted outputs, shape: (n_steps, output_size)
        """
         # [1, seq_len, input_size]

        # if externals_for_calibration is not None:
        #     calibration_window = torch.cat([calibration_window, externals_for_calibration], dim=1)

        

        if externals_for_calibration is not None:
            calibration_window = torch.cat([calibration_window, externals_for_calibration], dim=1) 
        

        calibration_window = calibration_window.unsqueeze(0)

        h0 = self.initial_hidden_state.expand(-1, 1, -1).contiguous()
        # calibration_window_inputs_only = calibration_window[:,:-self.ext_vars] if self.ext_vars is not None else calibration_window
        

        forward_outputs = self.forward(input_sequence=calibration_window, initial_states=h0)
        output_seq = forward_outputs.output_sequence.squeeze(0)
        next_input = output_seq[-1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, output_size]

        h_next = forward_outputs.final_hidden_state
        future_outputs = torch.zeros(n_steps, self.output_size, device=self.device)

        with torch.no_grad():
            for future_t in range(n_steps):
                if externals_for_forecasts is not None:
                    ext_t = externals_for_forecasts[future_t].view(1, 1, -1)  # [1, 1, ext_dim]
                    model_input = torch.cat([next_input, ext_t], dim=2)
                else:
                    model_input = next_input

                step_out = self.forward(model_input, h_next) #here is mdodel_input is already a concatenation of input_seq and  ext_t
                pred = step_out.output_sequence.squeeze(0).squeeze(0)
                future_outputs[future_t] = pred

                next_input = pred.unsqueeze(0).unsqueeze(0)
                h_next = step_out.final_hidden_state

        return output_seq, future_outputs

    def save_checkpoint(self, epoch: int, loss: float, optimizer: torch.optim.Optimizer,
                        checkpoint_dir: str = "checkpoints", cleanup: bool = False, add_stuffs:Optional[str]="") -> None:

        if cleanup:
            patterns = [
        os.path.join(checkpoint_dir, f"{self.name}{add_stuffs}_epoch*.pth"),
        os.path.join(checkpoint_dir, f"*{add_stuffs}_.csv")]
            
        for pattern in patterns:
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
                print(f"🗑️ Removed old checkpoint: {f}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"{self.name}{add_stuffs}_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, path)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            print(f"Checkpoint loaded from {checkpoint_path}. Epoch: {epoch}, Loss: {loss}")
            return epoch, loss
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
