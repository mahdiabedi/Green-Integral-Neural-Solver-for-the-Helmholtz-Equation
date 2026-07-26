#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

M.M.Abedi 2026
"""
import tensorflow as tf
import numpy as np
from scipy.special import hankel1,hankel2  
import matplotlib.pyplot as plt
import scipy.interpolate

import scipy.io

def interpolator(v, domain_bounds, xz, dtype=tf.float32):
    """
    Interpolate 2D array v over given xz locations using linear interpolation.

    Parameters:
    - v: 2D numpy array to interpolate
    - domain_bounds: tuple (a_x, b_x, a_z, b_z)
    - xz: query points of shape [N, 2]
    - dtype: TensorFlow data type

    Returns:
    - v_interpolated: interpolated values as tf.Tensor of shape [N, 1]
    """
    a_x,b_x,a_z,b_z=domain_bounds
    nz,nx=np.shape(v)
    # Create a grid of the original coordinates for v
    x_orig = np.linspace(a_x, b_x, nx)
    z_orig = np.linspace(a_z, b_z, nz)
    X_orig, Z_orig = np.meshgrid(x_orig, z_orig)
    
    # Flatten the grid and v
    points_orig = np.column_stack([X_orig.ravel(), Z_orig.ravel()])
    v_flat = v.ravel()
    
    # Interpolate using scipy's griddata
    v_interpolated = scipy.interpolate.griddata(points_orig, v_flat, xz, method='linear')

    # Convert to TensorFlow tensor 
    v_interpolated = tf.convert_to_tensor(v_interpolated, dtype=dtype)
    v_interpolated = tf.reshape(v_interpolated, (-1, 1))
    return v_interpolated


# Define the background wavefield U0 in 2D
def compute_U0(xz, s_xz, v0, omega,factor=1.):
    """
    Compute the background wavefield U0 for the 2D Helmholtz equation.
    
    Args:
    xz: Tensor of spatial coordinates.
    sx, sz: Source location (scalars).
    v0: Constant background velocity.
    omega: Angular frequency.
    factor: obtained by matching the analytical and finite-difference magnitudes
    
    Returns:
    U0: The background wavefield.
    """
    x, z = tf.unstack(xz, axis=-1)  # x and z will have shape [batch_size]
    x = tf.reshape(x, (-1, 1))  # Shape [batch_size, 1]
    z = tf.reshape(z, (-1, 1))  # Shape [batch_size, 1]
    sx, sz = tf.unstack(s_xz, axis=-1)
    # Compute the distance between the point and the source
    r = tf.sqrt((x - sx)**2 + (z - sz)**2)

    # Compute the argument for the Hankel function
    # Avoid division by zero by assigning a specific value when r is zero
    arg = tf.where(r == 0, tf.constant(1e-9, dtype=r.dtype), omega * r / v0)
    
    # Compute the background wavefield U0
    U0 = factor*(1j / 4) *hankel2(0,arg)
    U0_real = tf.math.real(U0)  # Shape: [batch_size, 1]
    U0_imag = tf.math.imag(U0)  # Shape: [batch_size, 1]
    U0=tf.concat([U0_real,U0_imag],axis=-1)
    # print('U0 stack real and imaginary:',np.shape(U0))
    return U0

def save_model_and_history(epoch, formatted_time, u_model, Loss, Error_val, parameters,file_name='training_history',folder_path='Results/'):
    print('saving...')
    #Update only the training time
    parameters["Training_time"] = formatted_time

    # Dynamically include the epoch number in the model filename
    model_filename = folder_path+f'Models/u_model_epoch_{epoch}.keras'

    # Save the model and training history
    u_model.save(model_filename)
    history = {
        "training_loss": Loss,
        "validation_error": Error_val,
        "parameters": parameters}
    np.save(folder_path +file_name+'.npy', history)
    print('\rSaved!       ')
    
    
def sin_activation(x):
    return tf.sin(x)


def next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def choose_pad_shape(nz, nx):
    # pad to next power of 2 in each dimension for fft
    return next_pow2(nz*2), next_pow2(nx*2)

def build_r_grid(nz, nx, dz, dx):
    """Return radial distance grid r[i,j] from the FFT origin (0,0) centered at (0,0)."""
    z = tf.cast(tf.range(-(nz/2.), nz - nz/2., dtype=tf.float64) * tf.cast(dz, tf.float64), dtype=tf.float32) 
    x = tf.cast(tf.range(-(nx/2.), nx - nx/2., dtype=tf.float64) * tf.cast(dx, tf.float64), dtype=tf.float32) 
    Z, X = tf.meshgrid(z, x, indexing='ij')
    r = tf.sqrt(Z*Z + X*X)
    return r, Z, X

def build_G0_kernel_2D_numpy(nx, nz, dx, dz, v0, omega,pad_to_pow2=True):

    # optionally pad to power-of-2
    if pad_to_pow2:
        Nz, Nx = choose_pad_shape(nz, nx)
    else:
        Nz, Nx = int(2*nz), int(2*nx)

    k0 = omega / v0

    # create 2D grid
    z = (np.arange(-Nz//2, Nz - Nz//2)) * dz
    x = (np.arange(-Nx//2, Nx - Nx//2)) * dx
    Z, X = np.meshgrid(z, x, indexing='ij')
    r = np.sqrt(Z**2 + X**2)

    # Regular Green's function away from r=0
    G = (1j / 4.0) * hankel2(0, k0 * np.maximum(r, 1e-12))

    # Cell-Averaged weak-singularity replacement at r=0 ----
    h = np.sqrt(dx * dz / np.pi)
    G_self = (1.0 / (2.0 * np.pi) * (np.log(h) + np.log(k0/2.0) + np.euler_gamma - 0.5)+ 1j / 4.0)

    # assign the self-interaction term
    G[r == 0.0] = G_self

    return G.astype(np.complex64)

def pad_and_decay(dm, w_damp,taper_type = "quadratic"):
    """
    Pad a 2-D tensor with width `w_damp` using nearest values,
    then exponentially (or by any other choices) damp the padded region.

    Parameters
    ----------
    dm : Input model. tf.Tensor or np.ndarray, shape (nz, nx)
    w_damp : Number of cells to pad on each side in z and x directions. (wz, wx)
    taper_type : quadratic,cosine,exponential,or constant

    Returns
    -------
    dm_padded : tf.Tensor, shape (nz+2*wz, nx+2*wx)

    mask : Damping mask (1 inside domain, <1 in padded rim).
    """
    wz, wx = w_damp

    # nearest-value padding ---
    dm_pad = np.pad(dm,pad_width=((wz, wz), (wx, wx)),mode="edge")   
    
    dm_pad = tf.cast(dm_pad, tf.float32) 
    
    #build grid indices ---
    nz, nx = tf.shape(dm_pad)[0], tf.shape(dm_pad)[1]
    z = tf.range(nz, dtype=tf.float32)
    x = tf.range(nx, dtype=tf.float32)
    Z, X = tf.meshgrid(z, x, indexing='ij')

    # distance to nearest point of the *original* domain
    # zero inside, >0 in padding
    dist_z = tf.maximum(tf.maximum(0.0, tf.cast(wz, tf.float32) - Z),
                        tf.maximum(0.0, Z - (tf.cast(nz, tf.float32) - wz - 1)))
    dist_x = tf.maximum(tf.maximum(0.0, tf.cast(wx, tf.float32) - X),
                        tf.maximum(0.0, X - (tf.cast(nx, tf.float32) - wx - 1)))
    dist = tf.maximum(dist_x / tf.cast(wx, tf.float32),dist_z / tf.cast(wz, tf.float32))

    # different taper choices ---
    # 1 inside interior, smooth decay to 0 at edge
    if taper_type == "quadratic":
        mask = 1.0 - dist**2   # simple quadratic decay (0 at edge, 1 inside)
    elif taper_type == "cosine":
        mask= 0.5 * (1 + tf.cos(np.pi * ( dist)))  # cos decay
    elif taper_type == "exponential":
        alpha = 6.0
        mask = tf.exp(-alpha * ( dist)**2)
    elif taper_type=="constant":
        mask=1
    else:
        raise ValueError("Unsupported taper_type")
    dm_pad=dm_pad*mask
    return dm_pad, mask

def sigmoid_beta_increase(step, num_epochs, beta_max=1.0, center_step=0.5, beta_rate=12.0):
    """
    An S-curve (Sigmoid) increase weight coefficient, dependent on the current epoch.
    
    Args:
        step: Current epoch.
        num_epochs: Total number of epochs for the training run.
        beta_max : The maximum coefficient value to reach.
        center_step : [0.0 to 1.0] indicating where the curve hits 50% of beta_max.
        beta_rate: Controls the transition slope. 
                           ~10 is a smooth curve. ~20 is sharp.
    """
    step = tf.cast(step, tf.float32)
    total = tf.cast(num_epochs, tf.float32)

    progress = step / total
    
    # Calculate scale-invariant sigmoid
    beta = beta_max / (1.0 + tf.exp(-beta_rate * (progress - center_step)))
    return beta

@tf.function()
def to_complex_grid(U_ri, nz, nx):
    """
    U_ri: Tensor or array shape (N,2) with N = nz*nx, order row-major (z major).
    Returns: complex tensor shape (nz, nx) with axis0 = z (rows), axis1 = x (cols).
    """
    U_ri = tf.convert_to_tensor(U_ri)
    Uc = tf.complex(U_ri[:,0], U_ri[:,1])   # flat complex (N,)
    return tf.reshape(Uc, [nz, nx])        # row-major reshape
@tf.function()
def from_complex_grid(Uc):
    """
    Uc: complex tensor shape (nz, nx) -> returns (N,2) with same row-major flatten.
    """
    Uc = tf.reshape(Uc, [-1])               # flatten in row-major
    return tf.stack([tf.math.real(Uc), tf.math.imag(Uc)], axis=-1)  # (N,2)

@tf.function()
def lisch_fft_apply(dm, U0, Us, omega, W, nz,nx, G0_kernel,w_damp=(0,0)):
    """
    Compute Lippmann-Schwinger Integral, hat{U}_s via FFT-based convolution on regular grids:
        hat{U}_s = - omega^2 * (G0 * f),  f = dm * (U0 + Us)
    where * denotes linear convolution on a uniform grid.

    Inputs:
      dm, U0, Us: complex64 tensors of shape [nz, nx]  (unpadded physical domain)
      G0_kernel: complex64 kernel [Nz, Nx], already padded and centered at (0,0)
      W=dx*dz: spacings
      omega: float
      w_damp: to remove the extra 2-D pad for damping with width `w_damp` 
    Returns:
      Uhat_s: complex64 tensor [nz, nx]
    """
    wz, wx = w_damp
    
    # 1) Source term
    f = dm * (U0+Us) * (-(omega**2) * W)
    f = to_complex_grid(f, nz, nx)


    Nz, Nx = G0_kernel.shape  # padded sizes

    # 2) Pad f to match padded size of G0
    pad_f = [[(Nz - nz)//2, Nz - nz - (Nz - nz)//2],
              [(Nx - nx)//2, Nx - nx - (Nx - nx)//2]]
    f_pad = tf.pad(f, paddings=pad_f)

    # 3) Shift kernel so impulse is at (0,0) for FFT convolution
    G_pad_shift = tf.signal.ifftshift(G0_kernel)

    # 4) FFTs
    Fk = tf.signal.fft2d(f_pad)
    Gk = tf.signal.fft2d(G_pad_shift)

    # 5) Multiply in Fourier domain and inverse FFT
    Uhat_pad_fft = Gk * Fk
    U_pad = tf.signal.ifft2d(Uhat_pad_fft) 

    # 6) Crop back to physical domain
    zs, ze = pad_f[0][0], pad_f[0][0] + nz
    xs, xe = pad_f[1][0], pad_f[1][0] + nx
    Uhat = U_pad[zs+wz:ze-wz, xs+wx:xe-wx]
    Uhat = from_complex_grid(Uhat)

    return Uhat # [nz*nx,2]



@tf.function()
def make_loss_GI(u_model,xz_LiSc,omega,v0,dm_LiSch,U0_LiSch,W,nx_LiSch,nz_LiSch,G0):
    
    Us_LiSch = u_model(xz_LiSc) 
    Uhat_s = lisch_fft_apply(dm_LiSch, U0_LiSch,Us_LiSch, omega, W, nz_LiSch, nx_LiSch, G0) 

    total_loss = tf.reduce_mean(tf.square(Us_LiSch-Uhat_s))

    return total_loss  


@tf.function()
def make_loss_PDE(u_model, U0, xz, v, v0, omega):
    # # tf.print("xz",tf.shape(xz))
    x, z = tf.split(xz, num_or_size_splits=2, axis=-1)  # x and z each have shape [batch_size, 1]
    # tf.print("x",tf.shape(x))
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, z])  # Watch both x and z
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, z])
            xz = tf.concat([x, z], axis=-1)  # Shape: [npts, 2]
            u = u_model(xz)  # Scattered wavefield (delta U), 2D outputs: real and imaginary

            # Split u into real and imaginary parts
            u_real = u[:, 0:1]  # Real part
            u_imag = u[:, 1:2]  # Imaginary part

        # Compute the first derivatives w.r.t both x and z for real and imaginary parts
        u_x_real,u_z_real = tape2.gradient(u_real, [x,z])  # First derivative w.r.t x (shape: [batch_size, 1])
        u_x_imag,u_z_imag = tape2.gradient(u_imag, [x,z])  # First derivative w.r.t x (imaginary)

    # Compute the second derivatives (Laplacian components) for real and imaginary parts
    u_xx_real = tape1.gradient(u_x_real, x)  # Second derivative w.r.t x
    u_zz_real = tape1.gradient(u_z_real, z)  # Second derivative w.r.t z
    u_xx_imag = tape1.gradient(u_x_imag, x)  # Second derivative w.r.t x (imaginary)
    u_zz_imag = tape1.gradient(u_z_imag, z)  # Second derivative w.r.t z (imaginary)

    # Compute the 2D Laplacian (sum of second derivatives w.r.t x and z) for both real and imaginary parts
    laplacian_u_real = u_xx_real + u_zz_real  # Real part of Laplacian
    laplacian_u_imag = u_xx_imag + u_zz_imag  # Imaginary part of Laplacian
    # Clean up tapes
    del tape1, tape2

    # Split real and imaginary parts of U0 
    U0_real = U0[:,0:1]  # Shape: [batch_size, 1]
    U0_imag = U0[:,1:2]  # Shape: [batch_size, 1]

    # Helmholtz equation residual for real and imaginary parts in 2D
    helmholtz_residual_real = omega**2 * (1 / v**2) * u_real + laplacian_u_real + omega**2 * (1 / v**2 - 1 / v0**2) * U0_real
    helmholtz_residual_imag = omega**2 * (1 / v**2) * u_imag + laplacian_u_imag + omega**2 * (1 / v**2 - 1 / v0**2) * U0_imag

    # Loss term from the Helmholtz equation residual for real and imaginary parts
    pde_loss_real =  tf.reduce_mean(tf.square(helmholtz_residual_real))
    pde_loss_imag = tf.reduce_mean(tf.square(helmholtz_residual_imag))

    # Total loss is the sum of both real and imaginary losses
    pde_loss = (pde_loss_real + pde_loss_imag)
    return pde_loss
    

#Plot real and imaginary parts of the wavefield
def plot_model_wavefield(wavefield, xz, npts_x, npts_z,domain_bounds,colormap='seismic',c_lims=None):
    a_x,b_x,a_z,b_z=domain_bounds
    # Extract the real and imaginary parts
    u_real = wavefield[:, 0]  # Real part
    u_imag = wavefield[:, 1]  # Imaginary part

    # Reshape the real part wavefield back into a 2D grid
    u_real_grid = tf.reshape(u_real, (npts_z, npts_x))  # Shape [npts_z, npts_x]
    
    # Reshape the imaginary part wavefield back into a 2D grid
    u_imag_grid = tf.reshape(u_imag, (npts_z, npts_x))  # Shape [npts_z, npts_x]

    # Plot the real part as a 2D image
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(u_real_grid, extent=[a_x, b_x, b_z, a_z], origin='upper', aspect='auto',cmap=colormap)
    if c_lims is not None:
        plt.clim(c_lims)
    plt.colorbar(label='Real Part')
    plt.title("Real Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    # Plot the imaginary part as a 2D image
    plt.subplot(122)
    plt.imshow(u_imag_grid, extent=[a_x, b_x,b_z, a_z], origin='upper', aspect='auto',cmap=colormap,interpolation='None')
    if c_lims is not None:
        plt.clim(c_lims)
    plt.colorbar(label='Imaginary Part')
    plt.title("Imaginary Part")
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    plt.tight_layout()
    