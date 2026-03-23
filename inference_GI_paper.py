#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Inference and Plotting Script for Paper: 
    A Green-integral--constrained neural solver with stochastic physics-informed regularization


This script loads pre-trained neural network models (GI, PINN, or hybrid GI+PDE) 
and evaluates their predicted scattered wavefields for given heterogeneous 
velocity models (e.g., Marmousi, Overthrust). 

Usage:
Modify the 'USER INPUT PARAMETERS' section below to select the desired 
frequency, velocity model, and training method. Ensure the corresponding 
data and model files are downloaded and extracted in their respective folders.

M.M.Abedi 2026
"""

import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# ============================================================================
#  USER INPUT PARAMETERS

# Source frequency in Hz (10 or 20)
frequency = 10        

# Target velocity model.  Available options: 'Marmousi', 'Overthrust', 'Otway'
velocity_model = 'Marmousi' 

# Neural solver training method used for the model.
# Available options: 
#   'GI'     : Green-Integral loss (Proposed method)
#   'GI+PDE' : Hybrid loss (Global integral + local PDE constraint)
#   'PINN'   : Standard Physics-Informed Neural Network (PDE + PML)
method = "GI" 


# ============================================================================
# DATA LOADING

model_path = f"PreTrained_Models/{velocity_model}_{frequency}_{method}.keras"
data_path  = f"data/xz_{velocity_model}_val.npz"

# ensure files exist before attempting to load them
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data not found: {data_path}")


# Load the validation grid and velocity data
data = np.load(data_path)
v_val = data['v_val']             # Velocity values on the regular grid
xz_val = data['xz_val']           # Spatial coordinates (x, z) arrays
npts_x_val = data['npts_x_val']   # Number of grid points in the x-direction
npts_z_val = data['npts_z_val']   # Number of grid points in the z-direction
extent = data['extent_original']  # Physical extent of the domain [xmin, xmax, zmin, zmax]

 #%% Plotting functions   
plt.rcParams.update({
    "text.usetex": True,           # Use LaTeX for text rendering
    "font.family": "serif",        # Set font family to serif
    "font.serif": ["Times"],       # Use Times as the serif font
    "font.size": 14,               # Set the default font size
    "axes.titlesize": 19,          # Title font size
    "axes.labelsize": 16,          # Label font size
    "xtick.labelsize": 14,         # x-tick label font size
    "ytick.labelsize": 14,         # y-tick label font size
    "legend.fontsize": 14,         # Legend font size
    "text.latex.preamble": r"\usepackage{amsmath}"  # Use amsmath for better LaTeX rendering
})

def sin_activation(x):
    return tf.sin(x)

def to_complex_grid(U_ri, nz, nx):
    """
    U_ri: Tensor or array shape (N,2) with N = nz*nx, order row-major (z major).
    Returns: complex tensor shape (nz, nx) with axis0 = z (rows), axis1 = x (cols).
    """
    U_ri = tf.convert_to_tensor(U_ri)
    Uc = tf.complex(U_ri[:,0], U_ri[:,1])   # flat complex (N,)
    return tf.reshape(Uc, [nz, nx])        # row-major reshape

# OLD!!! 
class EmbedderLayer(tf.keras.layers.Layer):#The sinusoidal encoder for K=3
    def __init__(self, domain_bounds, **kwargs):
        super(EmbedderLayer, self).__init__(**kwargs)
        self.domain_bounds = domain_bounds  # Store domain bounds for normalization

    @tf.function()
    def call(self, inputs):
        input1 =  (inputs)
        input2 = tf.math.multiply(input1 , 2.0)
        input4 = tf.math.multiply(input1 , 4.0)
        input8 = tf.math.multiply(input1 , 8.0)
        
        input_all = tf.concat([input1, input2, input4, input8], axis=1)
        # Apply sine and cosine functions
        sin_embed = tf.sin(input_all)
        cos_embed = tf.cos(input_all)

        # Concatenate original input, sine, and cosine embeddings
        output = tf.concat([inputs, sin_embed, cos_embed], axis=1)
        return output
    def get_config(self):
        config = super(EmbedderLayer, self).get_config()
        config.update({"domain_bounds": self.domain_bounds})
        return config

def model_prediction(model_path,x_in):
    u_model =keras.models.load_model(model_path,
        custom_objects={
            'EmbedderLayer': EmbedderLayer,
            'sin_activation': sin_activation}, compile=False)

    prediction = u_model(x_in)
    prediction_complex=to_complex_grid(prediction, npts_z_val, npts_x_val)
    
    # Extract real and imaginary parts
    u_real = prediction[:, 0]  # Real part
    u_real_grid = tf.reshape(u_real, (npts_z_val, npts_x_val)).numpy()
    u_imag = prediction[:, 1]  # imag part
    u_imag_grid = tf.reshape(u_imag, (npts_z_val, npts_x_val)).numpy()
    return u_real_grid,u_imag_grid,prediction_complex


#%% Plotting

#Velocity model
plt.figure(figsize=(6,4.5))
plt.imshow(tf.reshape(v_val, (npts_z_val, npts_x_val)), extent=extent, origin="upper", cmap="viridis", aspect="auto")
plt.title(f"{velocity_model} Velocity model")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
cbar = plt.colorbar(label='$v$ (normalized)', orientation='vertical')
cbar.ax.invert_yaxis()
plt.tight_layout()  
plt.show()

#Model prediction:
u_real_prediction,u_imag_prediction,prediction_complex=model_prediction(model_path,xz_val)

plt.figure(figsize=(12,5))
plt.suptitle(f"{method} Prediction for {velocity_model} model", fontsize=21, y=.98)

plt.subplot(1, 2, 1)
plt.imshow(u_real_prediction, extent=extent, origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.title("Real part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(u_imag_prediction, extent=extent, origin="upper",
           cmap="seismic", aspect="auto", interpolation='none')
plt.title("Imaginary part")
plt.ylabel("$z$ (km)")
plt.xlabel("$x$ (km)")
plt.colorbar()

plt.tight_layout()
plt.show()
