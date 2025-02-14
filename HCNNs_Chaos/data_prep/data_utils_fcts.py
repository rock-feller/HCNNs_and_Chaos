import torch
import numpy as np
import random
from typing import Tuple , Literal
from torch.utils.data import Dataset, DataLoader




class Normalization_Strategy():

    def __init__(self):

        self.data_normalization = True


    def Scale_ToNormalize(self, data :  torch.Tensor, 
                            scaling_factor: float) -> Tuple[np.ndarray , np.ndarray]:
        """
        Inputs:
                - the tensor data of shape (samples,features ) 
                - a scaling factor (float)
        
        Then, 
        Normalize the data  around 0 and multiply by the scaling factor according to the formula,
        scaled_data = scaling_factor*(data - data_averages)
        
        Outputs: 
                - the scaled data of shape  (samples,features,1) 
                - the average of the tensor data of shape (features,1)
        
                
        Inputs Shape: Tensor of shape (samples,features , 1) , float
        Outputs Shape: Tensor of shape(samples,features,1) , Tensor of shape (features,1)
        
        """

        tens_avgs = data.mean(axis=0)
        scaled_tens_data = scaling_factor*(data - tens_avgs) 

        
        return scaled_tens_data , tens_avgs

    def ScaleBackTo_originals(self, scaled_tens_data : torch.Tensor, scaling_factor: float ,
                           tens_avgs: torch.Tensor) ->  torch.Tensor:

        """This function takes as
        Inputs:
                - the scaled data of shape (samples,features,1) 
                - a scaling factor (float)
                - the average of the tensor data of shape (features,1)

        Then, 
        multiply the scaled data by the inverse of the scaling factor and
        add the averages along each axis
        
        
        Outputs: 
                - the original numpy array data of shape (samples,features)
                
        
        Inputs Shape: Tensor of shape (samples,features , 1) , float , Tensor of shape (features,1)
        Outputs Shape: Array of shape(samples,features,1) , Tensor of shape (samples , features)        
                
        """

        original_tens_data =  (scaled_tens_data / scaling_factor) + tens_avgs
        
        original_array_data  = original_tens_data.detach().cpu().numpy().squeeze()
        return original_array_data 




class Noisification_Strategy():
    def __init__(self ):
        self.add_noise = True


    def adding_sigmanoise_from_normal(self, chaos_trajs : torch.Tensor , sigma: float) -> torch.Tensor:
        """
        Inputs Shape: Tensor of shape (samples,features , 1) 
        Outputs Shape: Tensor of shape (samples,features , 1) 
        
        """
        noise_x =  np.random.normal(0,sigma,len(chaos_trajs))
        noise_y =  np.random.normal(0,sigma,len(chaos_trajs))
        noise_z =  np.random.normal(0,sigma,len(chaos_trajs))

        noisy_trajs = chaos_trajs + torch.tensor(np.vstack([noise_x,noise_y,noise_z]).T, 
                                                dtype= torch.float32).unsqueeze(2)

        return noisy_trajs
    

    def other_noise_strategy(self, chaos_trajs : torch.Tensor , sigma: float) -> torch.Tensor:
        return None
    




class SlidingWindowDataset(Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    def __init__(self, tensor_data, window_size):
        """
        Initializes the dataset with sliding windows.
        
        Inputs shape:
          - tensor_data: tensor of shape (sam[ples , features)
          - window_size: number of time steps to look back
  
        """
        self.data = tensor_data
        self.window_size = window_size
        self.slided_data = self.sliding_windows_shift_to(tensor_data, window_size)

    def sliding_windows_shift_to(self)-> torch.Tensor :#, tensor_data, window_size) -> torch.Tensor:
        """
        Creates sliding window from data.
        
        Args:
        data (torch.tensor): Input tensor of shape (m, 1, p)
        window_size (int): The size of the sliding window.
        
        Returns:
        torch.tensor: 4D tensor of shape (num_batches, window_size, 1, p)
        """
        num_batches = len(self.data) - self.window_size + 1
        x = []
        for i in range(num_batches):
            _x = self.data[i:i+self.window_size]
            x.append(_x)
        return torch.stack(x).float().to(self.device)  # Ensures that the output is double precision

    def __len__(self):
        return len(self.slided_data)

    def __getitem__(self, idx):
        return self.slided_data[idx]
    


    def contextwindow_testdata_generator(self, context_size :int, fcast_size:int,  location: Literal["random_", "last_"]):

        if location.lower() == 'random_':
            """
             Inputs Shape:
              -  context_size : int
             -  fcast_size: int 
              - location : str : random_  or last_ 
              
              Outputs Shape:

               - context_window: tensor of shape (context_size, features)
               - windowdata_toforecast: tensor of shape (fcast_size, features)
              """

            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            random_window_id =  random.randint( 0, full_data.size(0))
            context_window   = full_data[random_window_id,:context_size].squeeze(1)
            windowdata_toforecast = full_data[random_window_id,context_size:].squeeze(1)
            
            return context_window  , windowdata_toforecast

        elif  location.lower() == 'last_':
            """Here you must only pass in the entire dataset as a tensor of shape [n_samples, n_features]"""
            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            context_window   = full_data[-1,:context_size].squeeze(1)
            windowdata_toforecast = full_data[-1,context_size:].squeeze(1)
            return context_window  , windowdata_toforecast

        else:
            raise ValueError("s0_nature must be either 'random_' or 'last_' ")
        

class SlidingWindowDataLoader(DataLoader):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #dtype = torch.float32
    def __init__(self, dataset, batch_size=32, shuffle=False, device=device):
        """
        Custom DataLoader that loads batches from a dataset.
        
        Args:
        dataset (SlidingWindowDataset): The dataset to load from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data every epoch.
        device (torch.device): The device to map the data to.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle )
        #self.device = device

    def __iter__(self):
        """
        Iterate over batches and map data to the specified device.
        """
        batches = super().__iter__()
        for batch in batches:
            yield batch.to(self.device)




class InpTarg_TSDataset(Dataset):
    def __init__(self, data: torch.Tensor, window_size: int):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int):
        input_seq = self.data[idx:idx+self.window_size]
        target_seq = self.data[idx+1:idx+self.window_size+1]
        return input_seq, target_seq

    @classmethod
    def create_train_loader(cls, data: torch.Tensor, window_size: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        Inputs shape: 
            - tensor data of shaope (samples, features)
            - window_size: number of time steps to look back
            - batch_size: number of samples per batch
            - shuffle: whether to shuffle the data
        Outputs:
            - train_loader: DataLoader object : iterables over the dataset with inp_ and out_ batches of shape (batch_size, window_size, features)
        
        """
        train_dataset = cls(data, window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader
    


import torch
from torch.utils.data import Dataset, DataLoader
import random

class SlidingWindowDataset(Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    def __init__(self, data, window_size):
        """
        Initializes the dataset with sliding windows.
        
        Args:
        data (torch.tensor): Input tensor of shape (m, 1, p)
        window_size (int): The size of the sliding window.
        """
        self.data = data
        self.window_size = window_size
        self.slided_data = self.sliding_windows_shift_to(data, window_size)

    def sliding_windows_shift_to(self, data, window_size) -> torch.Tensor:
        """
        Creates sliding window from data.
        
        Args:
        data (torch.tensor): Input tensor of shape (m, 1, p)
        window_size (int): The size of the sliding window.
        
        Returns:
        torch.tensor: 4D tensor of shape (num_batches, window_size, 1, p)
        """
        num_batches = len(data) - window_size + 1
        x = []
        for i in range(num_batches):
            _x = data[i:i+window_size]
            x.append(_x)
        return torch.stack(x).float().to(self.device)  # Ensures that the output is double precision

    def __len__(self):
        return len(self.slided_data)

    def __getitem__(self, idx):
        return self.slided_data[idx]
    


    def contextwindow_testdata_generator(self, context_size :int, fcast_size:int,  location: Literal["random_", 'last_']):

        if location.lower() == 'random_':
            """Here you must only pass in the train dataset  as a tensor of shape [n_samples, n_features]"""

            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            random_window_id =  random.randint( 0, full_data.size(0))
            context_window   = full_data[random_window_id,:context_size].squeeze(1)
            windowdata_toforecast = full_data[random_window_id,context_size:].squeeze(1)
            
            return context_window  , windowdata_toforecast

        elif  location.lower() == 'last_':
            """Here you must only pass in the entire dataset as a tensor of shape [n_samples, n_features]"""
            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            context_window   = full_data[-1,:context_size].squeeze(1)
            windowdata_toforecast = full_data[-1,context_size:].squeeze(1)
            return context_window  , windowdata_toforecast

        else:
            raise ValueError("s0_nature must be either 'random_' or 'last_' ")