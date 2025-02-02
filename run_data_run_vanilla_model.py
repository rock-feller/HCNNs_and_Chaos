import pandas as pd
import torch
import matplotlib.pyplot as plt
from chaotic_data import systems, utils
lorenz_data  , _ =  systems.LorenzSolver(start=0. , stop=10.  , ics=(1.,  0. ,1.25), time_grid=0.01)
norma =  utils.Normalization_Strategy()
scaled_lorenz , avg_lorenz = norma.Scale_ToNormalize(data=lorenz_data, scaling_factor=0.02)
original_lorenz =  norma.ScaleBackTo_originals( scaled_tens_data=scaled_lorenz , scaling_factor=0.02 , tens_avgs=avg_lorenz)
train_size = 1500
train_lorenz =  scaled_lorenz[:train_size]
test_lorenz =  scaled_lorenz[train_size:]
data_sliding  =  utils.SlidingWindowDataset(data=train_lorenz, window_size=200)
sliced_data  =  data_sliding.sliding_windows_shift_to(train_lorenz,200)
context_data , toforecast_data = data_sliding.contextwindow_testdata_generator(context_size=500 , fcast_size=500, location='last_')
my_data_loader = utils.SlidingWindowDataLoader(dataset=sliced_data, batch_size=20, shuffle=False)


from HCNN.models import Vanilla_Model, PTF_Model , LForm_Model, LSpa_Model
import os
import torch
from torch import nn
from typing import Literal, Tuple, Optional

from torch.nn import MSELoss
loss_fct = MSELoss()
# Sample data
data_window = torch.randn(10, 5, 3)  # (batch_size, sequence_length, n_obs)
batch_size, seq_length, _ = data_window.size()
forecast_horizon = 500  # Predict 10 future time steps

# Initialize model, loss function, and optimizer
lform_model = LForm_Model(n_obs=3, n_hid_vars=10, s0_nature='random_', train_s0=False, batch_size=batch_size)


loss_fct = nn.MSELoss()
optimizer = torch.optim.Adam(lform_model.parameters(), lr=0.01)
losses = []
# Training loop
for epoch in range(3):
    for batch_of_data in my_data_loader:

        batch_size, seq_length, _ = batch_of_data.size()

        optimizer.zero_grad()

        # Run forward pass with forecast horizon
        predictions, states, delta_terms, forecasts, future_states = lform_model(batch_of_data)

        loss = loss_fct(predictions, batch_of_data) /batch_size
        loss.backward()
        optimizer.step()
        # print(f" Gradient of init_state : {hcnn_model.initial_hidden_state().grad}")
        # print(f" Gradient of A : {lform_model.cell.A.weight.data}")
        # print("\n")
        # print(f" Gradient of D : {lform_model.cell.D.weight.data}")
        losses.append(loss.item())
        with torch.no_grad():

            predictions, states, delta_terms, forecasts, future_states = lform_model(batch_of_data[-1].unsqueeze(0), forecast_horizon=forecast_horizon)

        # Compute loss on observed data

        # print(f" Gradient of init_state : {lspa_model.cell.Sparse_A.weight.data}")
    print(f"Epoch {epoch + 1}, Loss: {sum(losses)/len(losses)}")




for i in range(3):
    plt.figure(figsize=(24,5))
    plt.plot(forecasts[0,:,i] , 'r', label = 'pred')
    plt.plot(toforecast_data[:,i] , 'b', label = 'true')

    plt.legend()
plt.show()