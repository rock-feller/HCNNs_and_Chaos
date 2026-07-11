"""
Data Generation Utilities for Chaotic Systems

This module provides utilities for generating data from various chaotic systems
including Lorenz, Rössler, and other dynamical systems.
"""

import numpy as np
import torch
from scipy.integrate import odeint
from typing import Tuple, Optional, Literal, List, Callable


def _solve_ode(
    func: Callable,
    ics: Tuple[float, ...],
    start: float,
    stop: float,
    time_grid: float,
    burn_in: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Integrate an ODE and (optionally) discard the initial transient.

    Chaotic trajectories launched from an arbitrary initial condition spend some
    time off the attractor before settling onto it. Training on - and computing
    normalization statistics from - that transient contaminates the model. Set
    ``burn_in`` (in time units) to integrate ``burn_in`` extra time up front and
    drop it, so the returned trajectory lies on the attractor.

    Returns ``(trajectories[float32, (n_main, dim)], time_array[(n_main,)])`` where
    ``n_main == len(np.arange(start, stop, time_grid))``.
    """
    n_main = len(np.arange(start, stop, time_grid))
    n_burn = int(round(burn_in / time_grid)) if burn_in > 0 else 0
    total = n_burn + n_main
    time_full = start + np.arange(total) * time_grid
    states = odeint(func, list(ics), time_full)[n_burn:]
    time_array = np.arange(start, stop, time_grid)[:len(states)]
    return torch.tensor(states, dtype=torch.float32), time_array


def LorenzSolver(
    start: float,
    stop: float,
    ics: Tuple[float, float, float],
    time_grid: float,
    burn_in: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Self-contained Lorenz solver (sigma=10, rho=28, beta=8/3) with optional burn-in.

    Returns ``(trajectories, time_array)``.
    """
    def lorenz_equations(state, t):
        s, r, b = 10.0, 28.0, 8.0 / 3.0
        x, y, z = state
        return s * (y - x), r * x - y - x * z, x * y - b * z

    return _solve_ode(lorenz_equations, ics, start, stop, time_grid, burn_in)


class ChaoticSystemGenerator:
    """
    Generator for various chaotic dynamical systems.
    """
    
    def __init__(self):
        self.available_systems = ['lorenz', 'rossler', 'chua']
    
    def generate_lorenz(
        self,
        start: float = 0.0,
        stop: float = 20.0,
        time_grid: float = 0.01,
        initial_conditions: Tuple[float, float, float] = (1.0, 0.0, 1.25),
        parameters: Optional[dict] = None,
        burn_in: float = 0.0
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate Lorenz system trajectories.

        Args:
            start: Start time
            stop: Stop time
            time_grid: Time step
            initial_conditions: Initial conditions (x0, y0, z0)
            parameters: System parameters (sigma, rho, beta)
            burn_in: Transient time (in time units) to discard from the front so
                the returned trajectory lies on the attractor.

        Returns:
            Tuple of (trajectories, time_array)
            - trajectories: Shape (n_steps, 3)
            - time_array: Shape (n_steps,)
        """
        if parameters is None:
            parameters = {'sigma': 10, 'rho': 28, 'beta': 2.667}

        def lorenz_with_params(state, t):
            x, y, z = state
            sigma, rho, beta = parameters['sigma'], parameters['rho'], parameters['beta']
            x_dot = sigma * (y - x)
            y_dot = rho * x - y - x * z
            z_dot = x * y - beta * z
            return x_dot, y_dot, z_dot

        return _solve_ode(lorenz_with_params, initial_conditions,
                          start, stop, time_grid, burn_in)
    
    def generate_rossler(
        self,
        start: float = 0.0,
        stop: float = 100.0,
        time_grid: float = 0.01,
        initial_conditions: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        parameters: Optional[dict] = None,
        burn_in: float = 0.0
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate Rössler system trajectories.

        Args:
            start: Start time
            stop: Stop time
            time_grid: Time step
            initial_conditions: Initial conditions (x0, y0, z0)
            parameters: System parameters (a, b, c)
            burn_in: Transient time (in time units) to discard from the front.

        Returns:
            Tuple of (trajectories, time_array)
        """
        if parameters is None:
            parameters = {'a': 0.2, 'b': 0.2, 'c': 5.7}

        def rossler_equations(state, t):
            x, y, z = state
            a, b, c = parameters['a'], parameters['b'], parameters['c']
            x_dot = -y - z
            y_dot = x + a * y
            z_dot = b + z * (x - c)
            return x_dot, y_dot, z_dot

        return _solve_ode(rossler_equations, initial_conditions,
                          start, stop, time_grid, burn_in)
    
    def generate_chua(
        self,
        start: float = 0.0,
        stop: float = 100.0,
        time_grid: float = 0.01,
        initial_conditions: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        parameters: Optional[dict] = None,
        burn_in: float = 0.0
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate Chua's circuit trajectories.

        Args:
            start: Start time
            stop: Stop time
            time_grid: Time step
            initial_conditions: Initial conditions (x0, y0, z0)
            parameters: System parameters
            burn_in: Transient time (in time units) to discard from the front.

        Returns:
            Tuple of (trajectories, time_array)
        """
        if parameters is None:
            parameters = {'alpha': 15.6, 'beta': 28.0, 'm0': -1.143, 'm1': -0.714}

        def chua_equations(state, t):
            x, y, z = state
            alpha = parameters['alpha']
            beta = parameters['beta']
            m0, m1 = parameters['m0'], parameters['m1']

            # Chua's diode
            h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))

            x_dot = alpha * (y - x - h)
            y_dot = x - y + z
            z_dot = -beta * y
            return x_dot, y_dot, z_dot

        return _solve_ode(chua_equations, initial_conditions,
                          start, stop, time_grid, burn_in)
    
    def generate_system(
        self,
        system_name: Literal['lorenz', 'rossler', 'chua'],
        **kwargs
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate trajectories for specified chaotic system.
        
        Args:
            system_name: Name of the chaotic system
            **kwargs: System-specific parameters
            
        Returns:
            Tuple of (trajectories, time_array)
        """
        if system_name == 'lorenz':
            return self.generate_lorenz(**kwargs)
        elif system_name == 'rossler':
            return self.generate_rossler(**kwargs)
        elif system_name == 'chua':
            return self.generate_chua(**kwargs)
        else:
            raise ValueError(f"Unknown system: {system_name}. Available: {self.available_systems}")


def create_thesis_lorenz_dataset(
    total_time: float = 100.0,
    time_step: float = 0.04,
    initial_conditions: Tuple[float, float, float] = (0.0, 1.0, 1.05),
    system_parameters: Optional[dict] = None
) -> dict:
    """
    Create a Lorenz dataset following PhD thesis specifications.

    Args:
        total_time: Total simulation time (default 100.0 for richer dynamics)
        time_step: Integration time step (default 0.04 for thesis experiments)
        initial_conditions: Initial conditions (x0, y0, z0)
        system_parameters: Lorenz system parameters

    Returns:
        Dictionary containing trajectories, time array, and metadata
    """
    generator = ChaoticSystemGenerator()

    trajectories, time_array = generator.generate_lorenz(
        start=0.0,
        stop=total_time,
        time_grid=time_step,
        initial_conditions=initial_conditions,
        parameters=system_parameters
    )

    return {
        'trajectories': trajectories,
        'time_array': time_array,
        'metadata': {
            'system': 'lorenz',
            'total_time': total_time,
            'time_step': time_step,
            'n_steps': len(trajectories),
            'n_variables': trajectories.shape[1],
            'initial_conditions': initial_conditions,
            'parameters': system_parameters or {'sigma': 10, 'rho': 28, 'beta': 2.667},
            'thesis_compliant': True
        }
    }


def create_multi_system_dataset(
    systems: List[str],
    common_params: dict,
    system_specific_params: Optional[dict] = None
) -> dict:
    """
    Create datasets for multiple chaotic systems.
    
    Args:
        systems: List of system names
        common_params: Common parameters for all systems
        system_specific_params: System-specific parameters
        
    Returns:
        Dictionary with datasets for each system
    """
    generator = ChaoticSystemGenerator()
    datasets = {}
    
    if system_specific_params is None:
        system_specific_params = {}
    
    for system in systems:
        # Merge common and system-specific parameters
        params = {**common_params, **system_specific_params.get(system, {})}
        
        trajectories, time_array = generator.generate_system(system, **params)
        
        datasets[system] = {
            'trajectories': trajectories,
            'time_array': time_array,
            'metadata': {
                'system': system,
                'parameters': params,
                'n_steps': len(trajectories),
                'n_variables': trajectories.shape[1]
            }
        }
    
    return datasets


def generate_thesis_initial_conditions(
    n_samples: int = 60,
    x_range: Tuple[float, float] = (-10.0, 10.0),
    y_range: Tuple[float, float] = (-10.0, 10.0),
    z_range: Tuple[float, float] = (0.0, 30.0),
    seed: Optional[int] = 42
) -> List[Tuple[float, float, float]]:
    """
    Generate initial conditions following PhD thesis methodology.

    According to the thesis, 60 different initial conditions are sampled
    uniformly for each chaotic system to generate 100 different trajectories.

    Args:
        n_samples: Number of initial condition samples (default 60 as per thesis)
        x_range: Range for x0 values
        y_range: Range for y0 values
        z_range: Range for z0 values
        seed: Random seed for reproducibility

    Returns:
        List of initial condition tuples (x0, y0, z0)
    """
    if seed is not None:
        np.random.seed(seed)

    initial_conditions = []

    for _ in range(n_samples):
        x0 = np.random.uniform(x_range[0], x_range[1])
        y0 = np.random.uniform(y_range[0], y_range[1])
        z0 = np.random.uniform(z_range[0], z_range[1])
        initial_conditions.append((x0, y0, z0))

    return initial_conditions


def create_thesis_experiment_dataset(
    system_name: Literal['lorenz', 'rossler', 'chua'] = 'lorenz',
    n_trajectories: int = 100,
    total_time: float = 100.0,
    time_step: float = 0.04,
    train_ratio: float = 0.8,
    seed: Optional[int] = 42
) -> dict:
    """
    Create a complete experimental dataset following PhD thesis methodology.

    This generates multiple trajectories with different initial conditions
    as described in the thesis experimental setup.

    Args:
        system_name: Chaotic system to use
        n_trajectories: Number of trajectories to generate (default 100 as per thesis)
        total_time: Total simulation time for each trajectory
        time_step: Integration time step
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing multiple trajectories and metadata
    """
    # Generate initial conditions
    initial_conditions_list = generate_thesis_initial_conditions(
        n_samples=60, seed=seed
    )

    # Extend to 100 trajectories by reusing some initial conditions
    while len(initial_conditions_list) < n_trajectories:
        # Add slight perturbations to existing ICs
        base_ic = initial_conditions_list[len(initial_conditions_list) % 60]
        perturbed_ic = (
            base_ic[0] + np.random.normal(0, 0.1),
            base_ic[1] + np.random.normal(0, 0.1),
            base_ic[2] + np.random.normal(0, 0.1)
        )
        initial_conditions_list.append(perturbed_ic)

    generator = ChaoticSystemGenerator()
    trajectories_list = []

    print(f"Generating {n_trajectories} {system_name} trajectories...")

    for i, ic in enumerate(initial_conditions_list[:n_trajectories]):
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{n_trajectories} trajectories")

        trajectory, time_array = generator.generate_system(
            system_name,
            start=0.0,
            stop=total_time,
            time_grid=time_step,
            initial_conditions=ic
        )

        trajectories_list.append({
            'trajectory': trajectory,
            'initial_conditions': ic,
            'trajectory_id': i
        })

    # Split each trajectory into train/test
    train_trajectories = []
    test_trajectories = []

    for traj_data in trajectories_list:
        trajectory = traj_data['trajectory']
        split_idx = int(len(trajectory) * train_ratio)

        train_trajectories.append({
            'trajectory': trajectory[:split_idx],
            'initial_conditions': traj_data['initial_conditions'],
            'trajectory_id': traj_data['trajectory_id']
        })

        test_trajectories.append({
            'trajectory': trajectory[split_idx:],
            'initial_conditions': traj_data['initial_conditions'],
            'trajectory_id': traj_data['trajectory_id']
        })

    return {
        'train_trajectories': train_trajectories,
        'test_trajectories': test_trajectories,
        'time_array': time_array,
        'initial_conditions_list': initial_conditions_list,
        'metadata': {
            'system': system_name,
            'n_trajectories': n_trajectories,
            'total_time': total_time,
            'time_step': time_step,
            'train_ratio': train_ratio,
            'n_steps_per_trajectory': len(trajectory),
            'train_steps': split_idx,
            'test_steps': len(trajectory) - split_idx,
            'n_variables': trajectory.shape[1],
            'thesis_compliant': True,
            'seed': seed
        }
    }


# Convenience function for backward compatibility
def generate_lorenz_data(
    start: float = 0.0,
    stop: float = 20.0,
    time_grid: float = 0.01,
    ics: Tuple[float, float, float] = (1.0, 0.0, 1.25),
    burn_in: float = 0.0
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate Lorenz data using the original interface.

    Args:
        start: Start time
        stop: Stop time
        time_grid: Time step
        ics: Initial conditions
        burn_in: Transient time to discard from the front.

    Returns:
        Tuple of (trajectories, time_array)
    """
    return LorenzSolver(start, stop, ics, time_grid, burn_in)
