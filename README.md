# VAE Tutorial

A comprehensive tutorial on Variational Autoencoders (VAE) and Conditional Variational Autoencoders (CVAE) implemented in PyTorch.

An extension of https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f 

## Overview

This repository contains implementations and tutorials for:
- **Variational Autoencoders (VAE)**: A generative model that learns to encode data into a latent space and decode it back
- **Conditional Variational Autoencoders (CVAE)**: An extension of VAE that allows conditional generation based on labels

## Files

- `VAE.ipynb` - Complete tutorial and implementation of Variational Autoencoders
- `CVAE.ipynb` - Tutorial and implementation of Conditional Variational Autoencoders  
- `vae.py` - Standalone Python implementation of the VAE model class

## Features

- PyTorch implementation of VAE and CVAE
- Training on MNIST dataset
- Visualization of latent space and generated samples
- Support for both CPU and Apple Silicon (MPS) devices
- Comprehensive code examples and explanations

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- mpl_toolkits

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

1. Open the Jupyter notebooks to follow the tutorials:
   - Start with `VAE.ipynb` for basic VAE understanding
   - Move to `CVAE.ipynb` for conditional generation

2. Or use the standalone Python implementation:
   ```python
   from vae import VAE
   
   # Create model
   model = VAE(input_dim=784, latent_dim=2)
   ```

## Model Architecture

The VAE implementation includes:
- **Encoder**: Maps input data to latent space (mean and log-variance)
- **Reparameterization**: Samples from the latent distribution
- **Decoder**: Reconstructs data from latent representations

Default architecture:
- Input dimension: 784 (28x28 MNIST images)
- Hidden layers: 400 → 200 neurons
- Latent dimension: 2 (for visualization)

## Results

The models demonstrate:
- Effective reconstruction of MNIST digits
- Smooth interpolation in latent space
- Generation of new, realistic samples
- (CVAE) Conditional generation based on digit labels

## License

This project is for educational purposes.