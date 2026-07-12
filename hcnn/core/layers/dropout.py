"""
Custom dropout implementations for HCNN models.

This module provides specialized dropout layers including
partial teacher forcing dropout for HCNN training.
"""

import math
import torch
import torch.nn as nn



class PartialTeacherForcingDropout(nn.Dropout):
    r"""
    Implements dropout with scaling to enable partial teacher forcing.

    This layer applies dropout to the delta term (:math:`\hat{y} - y_{true}`) during state transitions
    to simulate partial teacher forcing, allowing for a controlled level of noise or guidance.

    Mathematical Formulation:
    -------------------------
    For input tensor :math:`\mathbf{x}` (typically the delta term: :math:`\hat{y} - y_{true}`):

    During training:

    .. math::
        \mathbf{y} = \frac{\mathbf{x} \odot \mathbf{m}}{1 - p}

    where :math:`\mathbf{m} \sim \text{Bernoulli}(1 - p)`, i.e., each element is:

    .. math::
        m_i = \begin{cases}
        1, & \text{with probability } (1 - p) \\
        0, & \text{with probability } p
        \end{cases}

    During evaluation:

    .. math::
        \mathbf{y} = \mathbf{x} \quad \text{(no dropout applied)}

    The scaling factor :math:`\frac{1}{1-p}` ensures that the expected value of the output
    matches the input: :math:`\mathbb{E}[\mathbf{y}] = \mathbb{E}[\mathbf{x}]` during training.

    The key difference from standard dropout is that this is specifically designed for
    the partial teacher forcing mechanism in HCNN models, where we want to randomly
    mask parts of the correction term during training.

    Parameters
    ----------
    p : float, default=0.0
        Dropout probability. The fraction of elements to drop
    inplace : bool, default=False
        Whether to perform the operation in-place

    Attributes
    ----------
    p : float
        Dropout probability. The fraction of elements to drop
    inplace : bool
        Whether to perform the operation in-place

    Examples
    --------
    >>> dropout = PartialTeacherForcingDropout(p=0.3)
    >>> delta_term = torch.randn(32, 3)  # batch_size=32, n_obs_vars=3
    >>> partial_delta = dropout(delta_term)
    >>> # During training, ~30% of elements will be zeroed out
    """

    def __init__(self, p: float = 0.0, inplace: bool = False):
        """
        Initialize the partial teacher forcing dropout layer.

        Parameters
        ----------
        p : float, default=0.0
            Dropout probability. Must be between 0.0 and 1.0
        inplace : bool, default=False
            Whether to perform the operation in-place

        Raises
        ------
        ValueError
            If p is not between 0.0 and 1.0
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
            
        super().__init__(p=p, inplace=inplace)
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply partial teacher forcing dropout to input tensor.

        During training, randomly sets elements to zero with probability `p`
        and scales the remaining elements by `1/(1-p)` to maintain expected value.
        During evaluation, returns input unchanged.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor (typically the delta term: y_true - y_pred)

        Returns
        -------
        torch.Tensor
            Tensor with dropout applied during training, unchanged during evaluation
        """
        return super().forward(input)

    def extra_repr(self) -> str:
        """Return extra representation string for the layer."""
        return f'p={self.p}, inplace={self.inplace}'


def create_ptf_dropout(prob: float) -> PartialTeacherForcingDropout:
    """
    Factory function to create a partial teacher forcing dropout module.

    This is a convenience function that matches the interface used in the original
    HCNN implementation for creating dropout modules.

    Parameters
    ----------
    prob : float
        Dropout probability for partial teacher forcing. Determines the fraction 
        of elements in the delta term tensor (`y_hat - y_true`) that are randomly 
        dropped (set to 0)

    Returns
    -------
    PartialTeacherForcingDropout
        A partial teacher forcing dropout module initialized with the given probability

    Examples
    --------
    >>> ptf_dropout = create_ptf_dropout(0.2)
    >>> delta_term = torch.randn(10, 3)
    >>> partial_delta = ptf_dropout(delta_term)
    """
    return PartialTeacherForcingDropout(p=prob)


class AdaptiveDropout(PartialTeacherForcingDropout):
    r"""
    Adaptive dropout for HCNN with partial teacher forcing.

    Implements the specific adaptive dropout strategy where:
    - First half of training: dropout probability = 0
    - Second half of training: dropout probability increases incrementally
      from 0 to target_p over the remaining epochs

    Mathematical Formulation:
    -------------------------
    Let :math:`N` = total_epochs, :math:`p_{target}` = target_p, :math:`e` = current_epoch

    .. math::
        p(e) = \begin{cases}
        0, & \text{if } e \leq \lfloor N/2 \rfloor \\
        p_{target} \cdot \frac{e - \lfloor N/2 \rfloor}{N - \lfloor N/2 \rfloor}, & \text{if } e > \lfloor N/2 \rfloor
        \end{cases}

    Simplified for second half:

    .. math::
        p(e) = p_{target} \cdot \frac{e - \lfloor N/2 \rfloor}{\lfloor N/2 \rfloor} \quad \text{for } e > \lfloor N/2 \rfloor

    Example with :math:`p_{target} = 0.25`, :math:`N = 10`:

    - Epochs 1-5: :math:`p = 0.0`
    - Epoch 6: :math:`p = 0.25 \cdot \frac{6-5}{5} = 0.25 \cdot \frac{1}{5} = 0.05`
    - Epoch 7: :math:`p = 0.25 \cdot \frac{7-5}{5} = 0.25 \cdot \frac{2}{5} = 0.10`
    - Epoch 8: :math:`p = 0.25 \cdot \frac{8-5}{5} = 0.25 \cdot \frac{3}{5} = 0.15`
    - Epoch 9: :math:`p = 0.25 \cdot \frac{9-5}{5} = 0.25 \cdot \frac{4}{5} = 0.20`
    - Epoch 10: :math:`p = 0.25 \cdot \frac{10-5}{5} = 0.25 \cdot \frac{5}{5} = 0.25`

    Parameters
    ----------
    target_p : float
        Target dropout probability to reach at the end of training
    total_epochs : int
        Total number of training epochs
    inplace : bool, default=False
        Whether to perform operations in-place
    """

    def __init__(
        self,
        target_p: float,
        total_epochs: int,
        inplace: bool = False
    ):
        super().__init__(p=0.0, inplace=inplace)
        self.target_p = target_p
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """
        Update the dropout probability based on current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based)
        """
        self.current_epoch = epoch

        # First half of training: dropout = 0
        if epoch <= self.total_epochs // 2:
            new_p = 0.0
        else:
            # Second half: incremental increase
            epochs_in_second_half = epoch - (self.total_epochs // 2)
            total_second_half_epochs = self.total_epochs - (self.total_epochs // 2)

            # Calculate incremental probability: target_p * (current_step / total_steps)
            new_p = self.target_p * (epochs_in_second_half / total_second_half_epochs)

        self.p = new_p

    def get_dropout_prob(self) -> float:
        """Get the current dropout probability."""
        return self.p

    def get_current_epoch(self) -> int:
        """Get the current epoch."""
        return self.current_epoch

    def extra_repr(self) -> str:
        """Return extra representation string for the layer."""
        return (
            f'target_p={self.target_p}, total_epochs={self.total_epochs}, '
            f'current_p={self.p:.3f}, '
            f'current_epoch={self.current_epoch}, '
            f'inplace={self.inplace}'
        )


class LinearScheduleDropout(PartialTeacherForcingDropout):
    r"""
    Linear schedule dropout that increases linearly from start_p to end_p.

    Mathematical Formulation:
    -------------------------
    Let :math:`N` = total_epochs, :math:`p_{start}` = start_p, :math:`p_{end}` = end_p, :math:`e` = current_epoch

    .. math::
        \text{progress} = \min\left(\frac{e}{N}, 1.0\right)

    .. math::
        p(e) = p_{start} + (p_{end} - p_{start}) \cdot \text{progress}

    This creates a linear interpolation between :math:`p_{start}` and :math:`p_{end}`:

    - At epoch 1: :math:`p \approx p_{start}`
    - At epoch :math:`N`: :math:`p = p_{end}`
    - Linear progression in between

    Example with :math:`p_{start} = 0.0`, :math:`p_{end} = 0.3`, :math:`N = 10`:

    - Epoch 1: :math:`p = 0.0 + (0.3 - 0.0) \cdot \frac{1}{10} = 0.03`
    - Epoch 5: :math:`p = 0.0 + (0.3 - 0.0) \cdot \frac{5}{10} = 0.15`
    - Epoch 10: :math:`p = 0.0 + (0.3 - 0.0) \cdot \frac{10}{10} = 0.30`

    Parameters
    ----------
    start_p : float
        Starting dropout probability
    end_p : float
        Ending dropout probability
    total_epochs : int
        Total number of training epochs
    inplace : bool, default=False
        Whether to perform operations in-place
    """

    def __init__(
        self,
        start_p: float,
        end_p: float,
        total_epochs: int,
        inplace: bool = False
    ):
        super().__init__(p=start_p, inplace=inplace)
        self.start_p = start_p
        self.end_p = end_p
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update dropout probability linearly."""
        self.current_epoch = epoch

        # Linear interpolation
        progress = min(epoch / self.total_epochs, 1.0)
        new_p = self.start_p + (self.end_p - self.start_p) * progress

        self.p = new_p

    def get_dropout_prob(self) -> float:
        return self.p

    def extra_repr(self) -> str:
        return (
            f'start_p={self.start_p}, end_p={self.end_p}, '
            f'total_epochs={self.total_epochs}, '
            f'current_p={self.p:.3f}'
        )


class ExponentialScheduleDropout(PartialTeacherForcingDropout):
    r"""
    Exponential schedule dropout that changes exponentially.

    Mathematical Formulation:
    -------------------------
    Let :math:`N` = total_epochs, :math:`p_{start}` = start_p, :math:`p_{end}` = end_p,
    :math:`\lambda` = decay_rate, :math:`e` = current_epoch

    .. math::
        \text{progress} = \min\left(\frac{e}{N}, 1.0\right)

    For exponential increase (:math:`p_{start} < p_{end}`):

    .. math::
        p(e) = p_{start} + (p_{end} - p_{start}) \cdot \left(1 - e^{-\lambda \cdot \text{progress} \cdot 10}\right)

    For exponential decrease (:math:`p_{start} > p_{end}`):

    .. math::
        p(e) = p_{end} + (p_{start} - p_{end}) \cdot e^{-\lambda \cdot \text{progress} \cdot 10}

    The factor of 10 scales the exponential to provide reasonable behavior
    over the training period. Higher decay_rate :math:`\lambda` leads to faster transitions.

    Example with :math:`p_{start} = 0.0`, :math:`p_{end} = 0.3`, :math:`\lambda = 0.1`, :math:`N = 10`:

    - Early epochs: slow increase from 0.0
    - Later epochs: approaches 0.3 asymptotically
    - The curve shape depends on :math:`\lambda`

    Parameters
    ----------
    start_p : float
        Starting dropout probability
    end_p : float
        Ending dropout probability
    total_epochs : int
        Total number of training epochs
    decay_rate : float, default=0.1
        Exponential decay rate
    inplace : bool, default=False
        Whether to perform operations in-place
    """

    def __init__(
        self,
        start_p: float,
        end_p: float,
        total_epochs: int,
        decay_rate: float = 0.1,
        inplace: bool = False
    ):
        super().__init__(p=start_p, inplace=inplace)
        self.start_p = start_p
        self.end_p = end_p
        self.total_epochs = total_epochs
        self.decay_rate = decay_rate
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update dropout probability exponentially."""
        self.current_epoch = epoch

        # Exponential schedule
        progress = min(epoch / self.total_epochs, 1.0)
        if self.start_p < self.end_p:
            # Exponential increase
            new_p = self.start_p + (self.end_p - self.start_p) * (1 - math.exp(-self.decay_rate * progress * 10))
        else:
            # Exponential decrease
            new_p = self.end_p + (self.start_p - self.end_p) * math.exp(-self.decay_rate * progress * 10)

        self.p = new_p

    def get_dropout_prob(self) -> float:
        return self.p

    def extra_repr(self) -> str:
        return (
            f'start_p={self.start_p}, end_p={self.end_p}, '
            f'decay_rate={self.decay_rate}, '
            f'total_epochs={self.total_epochs}, '
            f'current_p={self.p:.3f}'
        )


class CosineAnnealingDropout(PartialTeacherForcingDropout):
    r"""
    Cosine annealing dropout schedule.

    Mathematical Formulation:
    -------------------------
    Let :math:`N` = total_epochs, :math:`p_{start}` = start_p, :math:`p_{end}` = end_p, :math:`e` = current_epoch

    .. math::
        \text{progress} = \min\left(\frac{e}{N}, 1.0\right)

    .. math::
        p(e) = p_{end} + (p_{start} - p_{end}) \cdot \frac{1 + \cos(\pi \cdot \text{progress})}{2}

    This creates a smooth cosine transition from :math:`p_{start}` to :math:`p_{end}`:

    - At epoch 1: :math:`p \approx p_{start}` (since :math:`\cos(0) = 1`)
    - At epoch :math:`N/2`: :math:`p = \frac{p_{start} + p_{end}}{2}` (since :math:`\cos(\pi/2) = 0`)
    - At epoch :math:`N`: :math:`p = p_{end}` (since :math:`\cos(\pi) = -1`)

    The cosine function provides a smooth, non-linear transition that
    starts fast, slows in the middle, and speeds up again at the end.

    Example with :math:`p_{start} = 0.3`, :math:`p_{end} = 0.0`, :math:`N = 10`:

    - Epoch 1: :math:`p \approx 0.3` (starts high)
    - Epoch 5: :math:`p \approx 0.15` (middle value)
    - Epoch 10: :math:`p = 0.0` (ends at target)

    Parameters
    ----------
    start_p : float
        Starting dropout probability
    end_p : float
        Ending dropout probability
    total_epochs : int
        Total number of training epochs
    inplace : bool, default=False
        Whether to perform operations in-place
    """

    def __init__(
        self,
        start_p: float,
        end_p: float,
        total_epochs: int,
        inplace: bool = False
    ):
        super().__init__(p=start_p, inplace=inplace)
        self.start_p = start_p
        self.end_p = end_p
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update dropout probability using cosine annealing."""
        self.current_epoch = epoch

        # Cosine annealing
        progress = min(epoch / self.total_epochs, 1.0)
        new_p = self.end_p + (self.start_p - self.end_p) * (1 + math.cos(math.pi * progress)) / 2

        self.p = new_p

    def get_dropout_prob(self) -> float:
        return self.p

    def extra_repr(self) -> str:
        return (
            f'start_p={self.start_p}, end_p={self.end_p}, '
            f'total_epochs={self.total_epochs}, '
            f'current_p={self.p:.3f}'
        )


class StepScheduleDropout(PartialTeacherForcingDropout):
    r"""
    Step schedule dropout that changes at specific epochs.

    Mathematical Formulation:
    -------------------------
    Given a schedule :math:`S = \{(e_1, p_1), (e_2, p_2), \ldots, (e_k, p_k)\}`
    where :math:`e_i` are epoch thresholds and :math:`p_i` are dropout probabilities,
    sorted by epoch thresholds: :math:`e_1 \leq e_2 \leq \ldots \leq e_k`

    .. math::
        p(e) = p_i \quad \text{where } i = \min\{j : e \leq e_j\}

    In other words, find the first threshold :math:`e_i` such that the current
    epoch :math:`e` is less than or equal to :math:`e_i`, and use the corresponding :math:`p_i`.

    This creates a piecewise constant function with discrete jumps
    at the specified epoch thresholds:

    .. math::
        p(e) = \begin{cases}
        p_1, & \text{if } e \leq e_1 \\
        p_2, & \text{if } e_1 < e \leq e_2 \\
        \vdots & \\
        p_k, & \text{if } e_{k-1} < e \leq e_k
        \end{cases}

    Example with schedule = :math:`\{(5, 0.0), (10, 0.1), (\infty, 0.2)\}`:

    - Epochs 1-5: :math:`p = 0.0`
    - Epochs 6-10: :math:`p = 0.1`
    - Epochs 11+: :math:`p = 0.2`

    Parameters
    ----------
    schedule : list of tuples
        List of (epoch, dropout_prob) pairs
    inplace : bool, default=False
        Whether to perform operations in-place

    Examples
    --------
    >>> # Dropout = 0.0 for epochs 1-5, 0.1 for epochs 6-10, 0.2 for epochs 11+
    >>> schedule = [(5, 0.0), (10, 0.1), (float('inf'), 0.2)]
    >>> dropout = StepScheduleDropout(schedule)
    """

    def __init__(
        self,
        schedule: list,
        inplace: bool = False
    ):
        # Start with first probability
        initial_p = schedule[0][1] if schedule else 0.0
        super().__init__(p=initial_p, inplace=inplace)
        self.schedule = sorted(schedule)  # Sort by epoch
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update dropout probability based on step schedule."""
        self.current_epoch = epoch

        # Find the appropriate dropout probability for current epoch
        new_p = 0.0
        for epoch_threshold, prob in self.schedule:
            if epoch <= epoch_threshold:
                new_p = prob
                break

        self.p = new_p

    def get_dropout_prob(self) -> float:
        return self.p

    def extra_repr(self) -> str:
        return (
            f'schedule={self.schedule}, '
            f'current_p={self.p:.3f}'
        )


# Summary of available dropout strategies:
# 1. AdaptiveDropout: HCNN-specific (0 for first half, incremental for second half)
# 2. LinearScheduleDropout: Linear interpolation from start_p to end_p
# 3. ExponentialScheduleDropout: Exponential change with configurable decay rate
# 4. CosineAnnealingDropout: Cosine annealing schedule
# 5. StepScheduleDropout: Step-wise changes at specific epochs
# 6. PartialTeacherForcingDropout: Base dropout for delta terms (constant probability)
