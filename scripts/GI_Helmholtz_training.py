#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main training code for Paper: 
    A Green-integral--constrained neural solver with stochastic physics-informed regularization
M.M. Abedi 2026
"""

import tensorflow as tf
import numpy as np
import time
import os
from gi_solver.My_utilities_GI import make_loss_GI, make_loss_PDE, sigmoid_beta_increase, pad_and_decay, compute_U0, save_model_and_history, sin_activation, build_r_grid, build_G0_kernel_2D_numpy
from gi_solver.model import make_u_model 
from pathlib import Path

# Find the project root strictly for saving files 
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd() # Fallback for Jupyter
PROJECT_ROOT = current_dir.parent if current_dir.name == 'scripts' else current_dir

# Setup and save directories
os.makedirs(os.path.join(PROJECT_ROOT, 'Results', 'Models'), exist_ok=True)

dtype = "float32"
tf.keras.backend.set_floatx(dtype)

#%% User-defined input parameters =========-------------------==================
frequency =10        # Frequency in Hz 
neurons = 128        # Number of neurons in the hidden layers
neurons_final = 128  # Number of neurons in penultimate layer 
learning_rate=0.002
num_epochs=100000
npts =5000

# Target velocity model.  Available options: 'Marmousi', 'Overthrust', 'Otway'
velocity_model = 'Marmousi' 

# Neural solver training method used for the model.
# Available options: 
#   'GI'     : Green-Integral loss (Proposed method)
#   'GI+PDE' : Hybrid loss (Green integral + local PDE constraint)
method = "GI" 

validation_error = 'MAE'# 'MAE' or 'MSE' or False

use_lr_decay=True
plotting =True
seed=1234

#%% Load the validation data (and training collocation points for PDE loss)
# DATA LOADING

data_path_validation  = os.path.join(PROJECT_ROOT, f"data/data_{velocity_model}_validation.npz")

if not os.path.exists(data_path_validation):
    raise FileNotFoundError(f"Data not found: {data_path_validation}")


# Load the validation grid and velocity data
data = np.load(data_path_validation)
v_val = data['v_val']             # Velocity values on the regular grid
xz_val = data['xz_val']           # Spatial coordinates (x, z) arrays
dU_2d     = data['dU_2d']
npts_x_val = data['npts_x_val'].item()    # Number of grid points in the x-direction
npts_z_val = data['npts_z_val'] .item()   # Number of grid points in the z-direction
extent = data['extent_original'].tolist()  # Physical extent of the domain [xmin, xmax, zmin, zmax]
v0         = data['v0'].item() #backhground velocity
s_xz       = data['s_xz'] #source location
s_xz=tf.convert_to_tensor(s_xz)

a_x, b_x, b_z, a_z=extent
omega = np.float32(frequency*2*np.pi)  # Angular frequency
dU_2d = tf.cast(dU_2d, dtype=dtype)
if plotting:
    plt.figure(figsize=(10, 6))
    plt.scatter(xz_val[:, 0], xz_val[:, 1], c=v_val, cmap='viridis', s=5)
    plt.colorbar()
    
if method== 'GI+PDE':
    data_path_colocation=os.path.join(PROJECT_ROOT, f"data/{velocity_model}_random_training_importanceS.mat")
    mat_data = scipy.io.loadmat(data_path_colocation)

    v_all=mat_data['v_all']
    xz_all=mat_data['xz_all']
    n_all=xz_all.shape[0]
    v_all=tf.cast(v_all,dtype=dtype)
    xz_all=tf.cast(xz_all,dtype=dtype)
    print(f'{velocity_model} colocation points data is loaded.')


#Normalize to one wavelength:
L = 2.0 * np.pi * v0 / omega
xz_val = xz_val / L
v_val = v_val / L
s_xz = s_xz / L
v0 = v0 / L
a_x,b_x,a_z,b_z=a_x/L,b_x/L,a_z/L,b_z/L
if dtype=="float32":
    v0=np.float32(v0)
v_val=tf.cast(v_val,dtype=dtype)
shift_np = [(b_x+a_x) / 2, (b_z+a_z) / 2]
xz_val = tf.cast(xz_val - shift_np, dtype=dtype)
s_xz = tf.cast(s_xz - shift_np, dtype=dtype)

a_x, b_x = a_x - shift_np[0], b_x - shift_np[0]
a_z, b_z = a_z - shift_np[1], b_z - shift_np[1]
if method== 'GI+PDE':
    xz_all = xz_all / L
    v_all = v_all / L
    xz_all = tf.cast(xz_all - shift_np, dtype=dtype)
    U0_all = compute_U0(xz_all, s_xz, v0, omega)#calculate when chaning the colocation points
            
domain_bounds=a_x,b_x,a_z,b_z

#Schedule for the coefficient of the PDE loss
if method=="GI+PDE":
    if velocity_model=="Marmousi":
        beta_rate=20
        beta_max=1/100
    else:
        beta_rate=16
        beta_max=1/400   
    center_step=0.5    
else:
    beta_rate=np.nan
    beta_max=np.nan
    
#%% GI grid
data_path_GI  = os.path.join(PROJECT_ROOT, f"data/data_{velocity_model}_GIgrid.npz")

if os.path.exists(data_path_GI):
    
    # Load the GI grid of dm
    data = np.load(data_path_GI)
    dm_LiSch = data['dm_LiSch']             
    dx_scatterer = data['dx_scatterer'].item()       
    dz_scatterer = data['dz_scatterer'].item()
    nx_LiSch = data['nx_LiSch'].item()   
    nz_LiSch = data['nz_LiSch'].item()  
    w_damp = tuple(data['w_damp'].tolist())  

    dm_LiSch = tf.cast(dm_LiSch, dtype)
    print('GI grid data is loaded.')
else:

    nx_LiSch = npts_x_val
    nz_LiSch = npts_z_val

    #Quadrature weight:
    x = np.linspace(a_x + 0.5*(b_x- a_x) / (nx_LiSch),b_x - 0.5*(b_x - a_x) / (nx_LiSch), num=nx_LiSch)
    z = np.linspace(a_z + 0.5*(b_z - a_z) / (nz_LiSch),b_z- 0.5*(b_z - a_z) / (nz_LiSch),num=nz_LiSch)
    dx_scatterer=x[1]-x[0]#Here we only find the dx and dz. the grids are builts later
    dz_scatterer=z[1]-z[0]

    _,Z,X=build_r_grid(nz_LiSch, nx_LiSch, dz_scatterer, dx_scatterer)#The grids are built here
    x_flat = tf.reshape(X,(-1, 1))
    z_flat = tf.reshape(Z,(-1, 1))
    xz_LiSc= tf.concat((x_flat, z_flat), axis=-1)  # (npts, 2)
    
    #adding damping dm region around the model (tapering):
    w_damp = (max(10, int((v0/frequency) // dz_scatterer)), 
              max(10, int((v0/frequency) // dx_scatterer)))
    
    v_LiSch = scipy.interpolate.griddata(xz_val, v_val, xz_LiSc.numpy(), method='linear')
    mask = np.isnan(v_LiSch)  # outside the v_val use nearest neighbor
    v_LiSch[mask.ravel()] = scipy.interpolate.griddata(xz_val, v_val, xz_LiSc.numpy()[mask.ravel()], method='nearest')
    dm_LiSc = 1.0 / v_LiSch**2 - 1.0 / v0**2  # shape (npts,)
    v_LiSch= tf.cast(v_LiSch,dtype)
    dm_LiSch= tf.cast(dm_LiSc,dtype)
    print("w_damp = ",w_damp)
    
    dm_LiSch,mask=pad_and_decay(tf.cast(tf.reshape(dm_LiSc,(nz_LiSch,nx_LiSch)),dtype), w_damp,taper_type = "cosine")
    nz_LiSch, nx_LiSch=nz_LiSch+2*w_damp[0], nx_LiSch+2*w_damp[1]

    dm_LiSch=tf.reshape(dm_LiSch,(-1,1))
    
    
v_LiSch=tf.math.sqrt(1/(dm_LiSch + 1.0 / v0**2)) 
_,Z_LiSch,X_LiSch=build_r_grid(nz_LiSch, nx_LiSch, dz_scatterer, dx_scatterer)
x_flat = tf.reshape(X_LiSch,(-1, 1))
z_flat = tf.reshape(Z_LiSch,(-1, 1))
dx_scatterer=X_LiSch[0,1]-X_LiSch[0,0]#scatterer
dz_scatterer=Z_LiSch[1,0]-Z_LiSch[0,0]
W=tf.cast(dx_scatterer*dz_scatterer,dtype)
xz_LiSc= tf.concat((x_flat, z_flat), axis=-1)  # (npts, 2)
domain_bonds_LiSch=np.min(X_LiSch),np.max(X_LiSch),np.min(Z_LiSch),np.max(Z_LiSch)
#end of tapering
print("nx_LiSch * nx_LiSch",nx_LiSch*nz_LiSch)
print('W=',np.float32(W))


#U0 precomputing   
U0_LiSch=compute_U0(xz_LiSc, s_xz, v0, omega)


G0 = build_G0_kernel_2D_numpy(nx_LiSch, nz_LiSch, dx_scatterer, dz_scatterer, v0=tf.constant(v0, tf.float32), omega=tf.constant(omega, tf.float32),pad_to_pow2=True)
G0 = tf.constant(G0)

NzG0, NxG0 = G0.shape  # padded sizes
# Shift kernel so impulse is at (0,0) for FFT convolution
G_shift = tf.signal.ifftshift(G0)
Gk = tf.signal.fft2d(G_shift)

#%% Training config
@tf.function()
def train_step_GI(u_model,optimizer,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,G0,nx_LiSch,nz_LiSch,xz_val,cal_error=False,dU_2d_val=0.):
                
    with tf.GradientTape() as tape:
        Loss=make_loss_GI (u_model,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,W, nx_LiSch,nz_LiSch,G0)
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    del tape
    
    if cal_error == 'MAE':
        u=u_model(xz_val)
        error = tf.reduce_mean(tf.abs(dU_2d_val - u))
    elif cal_error == 'MSE': 
        u=u_model(xz_val)
        error = tf.reduce_mean(tf.square(dU_2d_val - u))
    else:
        error=0.
    return Loss,error

@tf.function()
def train_step_GIPDE(u_model,optimizer,U0, xz, v,beta_PDE,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,G0,nx_LiSch,nz_LiSch,xz_val,cal_error=False,dU_2d_val=0.):
                
    with tf.GradientTape() as tape:
        Loss_GI=make_loss_GI (u_model,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,W, nx_LiSch,nz_LiSch,G0)
        Loss_PDE = make_loss_PDE (u_model, U0, xz, v, v0, omega)
        Loss=Loss_GI+beta_PDE*Loss_PDE
        
    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    del tape
    
    if cal_error == 'MAE':
        u=u_model(xz_val)
        error = tf.reduce_mean(tf.abs(dU_2d_val - u))
    elif cal_error == 'MSE': 
        u=u_model(xz_val)
        error = tf.reduce_mean(tf.square(dU_2d_val - u))
    else:
        error=0.
    return Loss,error
 
    
# Define the optimizer
if use_lr_decay:
    initial_learning_rate=learning_rate
    decay_steps=10000
    decay_rate=0.9
    final_learning_rate = initial_learning_rate * (decay_rate ** (num_epochs / decay_steps))
    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate)
    
optimizer =keras.optimizers.Adam(learning_rate=learning_rate)  

u_model = make_u_model(neurons,activation=sin_activation,neurons_final=neurons_final,seed=seed,domain_bounds=domain_bounds)

optimizer.build(u_model.trainable_variables)
u_model.summary()

#%% TRAINING LOOP <<<<<<<<<<<<<<<<<<<
#!!!!

Loss= []
Loss_val = []
Error_val = []
gpus = tf.config.list_physical_devices('GPU')
gpu_details = []

#Define parameters saved in history
parameters = {
    "velocity_model":velocity_model,
    "omega": float(omega),
    "v0": v0,
    "neurons": (neurons,neurons_final),
    "activation": sin_activation,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "domain_bounds":domain_bounds,
    "Source_xz": list(s_xz.numpy()),
    "npts": npts,
    "seed":seed,
    "validation_error_type":validation_error,
    "use_GI":{method,w_damp,(nz_LiSch, nx_LiSch),(beta_rate,beta_max)},
    "GPU":gpu_details}   
print(parameters)

start_time = time.time()
epoch_time=start_time

#training loop:
for epoch in range(num_epochs):

    cal_error=False
    compute_validation=False
    if epoch%100==0:
        cal_error=validation_error
        
    if method == "GI+PDE" :
        rng = np.random.default_rng(seed=epoch)  # Set the seed here
        random_indices = np.sort(rng.choice(n_all, npts, replace=False))
        xz_train = tf.gather(xz_all, random_indices)  # Gather random colocation points
        v_train = tf.gather(v_all, random_indices)  # Gather corresponding v
        U0_train = tf.gather(U0_all, random_indices)  # Gather corresponding U0 
        beta_PDE=sigmoid_beta_increase(epoch, beta_max=beta_max, center_step=center_step, beta_rate=beta_rate)#coefficient of PDE_loss
        loss_train,error_val =  train_step_GIPDE(u_model,optimizer,U0_train, xz_train, v_train,beta_PDE,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,G0,nx_LiSch,nz_LiSch,xz_val,cal_error=cal_error,dU_2d_val=dU_2d)

    
    if method == "GI" :
         loss_train,error_val =  train_step_GI(u_model,optimizer,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,G0,nx_LiSch,nz_LiSch,xz_val,cal_error=cal_error,dU_2d_val=dU_2d)

    Loss.append(loss_train)
    if cal_error:
        Error_val.append(error_val)
        
    # Check if loss_train is NaN
    if tf.math.is_nan(loss_train):
        print(f"Stopping training due to NaN loss at epoch {epoch}")
        break

    if epoch%100==0:
        print(" Epoch %d of %d" % (int(epoch), int(num_epochs)), end='\n')
        print(" Loss: %.4e, Val Error: %.4f,  Time taken: %.2fs" 
              % (float(loss_train),float(error_val),time.time()-epoch_time),end='\n')
        epoch_time=time.time()
    if (epoch == 0 or epoch == 500 or epoch == 5000 or (epoch % 10000 == 0)):
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
        #saving during training
        save_model_and_history(epoch, formatted_time, u_model, Loss,Error_val,parameters,folder_path=os.path.join(PROJECT_ROOT, 'Results/'))

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
# Convert elapsed time to minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)
formatted_time = f'{int(minutes)} min {seconds:.0f} sec'
print(f'Training time: {formatted_time}')

#for recording GPU memory:
if gpus:
    for i, gpu in enumerate(gpus):
        # Get hardware info
        details = tf.config.experimental.get_device_details(gpu)
        
        # Get memory info (current + peak)
        try:
            mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            details.update({'memory_info': mem_info})
        except Exception as e:
            details.update({'memory_info': None, 'memory_error': str(e)})
        gpu_details.append(details)
    print("GPU Details:", gpu_details)
parameters["GPU"]=gpu_details

#% final saving<<<<<<<<<<<<<<<
save_model_and_history(epoch, formatted_time, u_model, Loss,Error_val,parameters,folder_path=os.path.join(PROJECT_ROOT, 'Results/'))

if plotting:
    plt.figure(figsize=(6, 4))
    plt.plot(Loss,'k')
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel(f'{method} Loss',fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
