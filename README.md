# Auto Encoders

This repo is a about Auto-Encoder examples collected in different notebook, for the main purpose of testing them for recommendations. And the work culminates in the RQ-VAE which follows this paper. All the Auto-Encoders are simple versions and the results could be quite alot better.

The first part of each notebook is a base version of Auto-Encoders using the MNIST dataset and the second part is trying to create a recommender with the certain technology based on the last-FM dataset.

## Auto Encoder

The `ae.ipynb` contains a standard denoising Auto-Encoder, with just 2 simple NNs.
The resulting NDCG@10 for a certain test set is 0.075.

## Variational Auto Encoder

The `vae.ipynb` contains a variational Auto-Encoder, trying to learn a Normal Distribution over the latent space.
The resulting NDCG@10 for a certain test set is 0.035.

## ResNet

## Residual Quantization

## VQ-VAE

## RQ-VAE
