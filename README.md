# Green-Integral-Neural-Solver-for-the-Helmholtz-Equation
This repository provides training code pre-trained models to simulate highly oscillatory scattered wavefields.

[![Work in Progress](https://img.shields.io/badge/Status-Under%20Construction-orange)](#status)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)

Official code repository for the paper: **"A Green-integral--constrained neural solver with stochastic physics-informed regularization"** by Mohammad Mahdi Abedi, David Pardo, and Tariq Alkhalifah.

## Overview

Standard Physics-Informed Neural Networks (PINNs) struggle to resolve highly oscillatory scattered wavefields in heterogeneous media. Relying purely on local PDE residuals can lead to non-physical solutions and requires computationally expensive boundary conditions (like PMLs). 

This repository implements a **Green-Integral (GI) neural solver** that overcomes these limitations. By enforcing global consistency through the Lippmann-Schwinger integral equation, the neural network inherently satisfies the Sommerfeld radiation condition, completely bypassing the need for artificial boundary layers and second-order spatial derivatives.

## Repository Structure

Currently, the repository contains the pre-trained models and inference code to reproduce the visual results from the paper.

* `PreTrained_Models/`: Contains the saved `.keras` neural network models for various configurations (GI, PINN, Hybrid) and velocity models (Marmousi, Overthrust, Otway).
* `data/`: Contains the `.npz` files with the validation grids and true velocity models required for plotting.
* `inference.py`: The script to load a pre-trained model and plot the predicted scattered wavefields.

This repository is being built. The complete training scripts including the FFT-accelerated GI loss and the hybrid GI+PDE implementations will be uploaded in the near future.

You can install the dependencies using:
```bash
pip install tensorflow numpy matplotlib
