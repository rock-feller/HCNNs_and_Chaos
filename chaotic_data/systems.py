import numpy as np
from scipy.integrate import odeint
from typing import Tuple
import torch



def lorenz(state , t):
    s=10
    r=28
    b=2.667
    x, y, z = state
    """
    Inputs:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    
    Then:
    Put together the lorenz equations in the form:

    x_dot : dx/dt = s(y - x)
    y_dot : dy/dt = rx - y - xz
    z_dot : dz/dt = xy - bz

    Outputs:
            - x_dot
            - y_dot
            - z_dot
    values of the lorenz attractor's partial derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return x_dot, y_dot, z_dot
  
def LorenzSolver(start :int, stop:int, ics : Tuple[float , float , float] ,
                 time_grid : float)-> Tuple[np.ndarray , np.ndarray , np.ndarray, np.ndarray]:
    
    """
    Inputs:
            -  start: the start time of the lorenz system : int
            -  stop: the stop time of the lorenz system : int
            - ics: initial conditions of the lorenz system 
            - time grid : the time grid of the lorenz system :
    [start and stop] correspond to the the range within which you want the data to be generated
    The larger the time grid , the less spaced the data points will be.
    The nber of points = (stop - start) / time_grid
    
    Then, 
    Solve the Lorenz system of equations for the given initial conditions and time range

    Outputs:
            - x values of the lorenz system
            - y values of the lorenz system
            - z values of the lorenz system
            - time values of the lorenz system
    
    Outputs shape: Tensor shape : (nber of points , 3, 1)
    """
    x0 , y0 , z0 = ics
    
    state0 = [x0 , y0 , z0]
    timegrid_array = np.arange(start, stop, time_grid)
    states = odeint(lorenz, state0, timegrid_array)
    xs = states[:,0]
    ys = states[:,1]
    zs = states[:,2]

    lor_trajs= torch.tensor(np.vstack([xs,ys,zs]).T , dtype= torch.float32)

    return lor_trajs, timegrid_array





def rossler_eqs(state, t):
    a = 0.2
    b = 0.2
    c = 5.7
    x, y, z = state
    """
    Inputs:
       x, y, z: a point of interest in three dimensional space
       a, b, c: parameters defining the Rössler attractor
    
    Then:
    Put together the Rössler equations in the form:

    x_dot : dx/dt = -y - z
    y_dot : dy/dt = x + ay
    z_dot : dz/dt = b + z(x - c)

    Outputs:
            - x_dot
            - y_dot
            - z_dot
    values of the Rössler attractor's partial derivatives at the point x, y, z
    """
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)

    return x_dot, y_dot, z_dot

def RosslerSolver(start: int, stop: int, ics: Tuple[float, float, float], time_grid: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Inputs:
            - start: the start time of the Rössler system : int
            - stop: the stop time of the Rössler system : int
            - ics: initial conditions of the Rössler system 
    [start and stop] correspond to the the range within which you want the data to be generated
    
    Then, 
    Solve the Rössler system of equations for the given initial conditions and time range

    Outputs:
            - x values of the Rössler system
            - y values of the Rössler system
            - z values of the Rössler system
            - time values of the Rössler system

     Outputs shape: Tensor shape : (nber of points , 3, 1)

    """
    x0, y0, z0 = ics
    
    state0 = [x0, y0, z0]
    state0 = [x0 , y0 , z0]
    timegrid_array = np.arange(start, stop, time_grid)
    states = odeint(rossler_eqs, state0, timegrid_array)
    xs = states[:,0]
    ys = states[:,1]
    zs = states[:,2]

    ross_trajs= torch.tensor(np.vstack([xs,ys,zs]).T , dtype= torch.float32)

    return ross_trajs, timegrid_array


def rabi_fabri_eqs(state, t):
    alpha = 0.98
    gamma = 0.1
    x, y, z = state
    """
    Inputs:
       x, y, z: a point of interest in three dimensional space
       alpha, gamma: parameters defining the Rabinovich-Fabrikant attractor
    
    Then:
    Put together the Rabinovich-Fabrikant equations in the form:

    x_dot : dx/dt = y(z - 1 + x^2) + gamma * x
    y_dot : dy/dt = x(3z + 1 - x^2) + gamma * y
    z_dot : dz/dt = -2z(alpha + xy)

    Outputs:
            - x_dot
            - y_dot
            - z_dot
    values of the Rabinovich-Fabrikant attractor's partial derivatives at the point x, y, z

   
    """
    x_dot = y * (z - 1 + x**2) + gamma * x
    y_dot = x * (3 * z + 1 - x**2) + gamma * y
    z_dot = -2 * z * (alpha + x * y)

    return x_dot, y_dot, z_dot

def RabinovichFabrikantSolver(start: int, stop: int, ics: Tuple[float, float, float], 
                              time_grid: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Inputs:
            - start: the start time of the Rabinovich-Fabrikant system : int
            - stop: the stop time of the Rabinovich-Fabrikant system : int
            - ics: initial conditions of the Rabinovich-Fabrikant system 
    [start and stop] correspond to the the range within which you want the data to be generated
    
    Then, 
    Solve the Rabinovich-Fabrikant system of equations for the given initial conditions and time range

    Outputs:
            - x values of the Rabinovich-Fabrikant system
            - y values of the Rabinovich-Fabrikant system
            - z values of the Rabinovich-Fabrikant system
            - time values of the Rabinovich-Fabrikant system
            
     Outputs shape: Tensor shape : (nber of points , 3, 1)
    """
    x0, y0, z0 = ics  # for chaotic trajs consider 0.1, 0.1, 0.1
    state0 = [x0, y0, z0]
    timegrid_array = np.arange(start, stop, time_grid)
    states = odeint(rabi_fabri_eqs, state0, timegrid_array)
    xs = states[:,0]
    ys = states[:,1]
    zs = states[:,2]

    rabifabri_trajs= torch.tensor(np.vstack([xs,ys,zs]).T , dtype= torch.float32)

    return rabifabri_trajs, timegrid_array
