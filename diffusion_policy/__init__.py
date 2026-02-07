"""
diffusion_policy/ — Core package for the Diffusion Policy project.
===================================================================

This package implements a Denoising Diffusion Probabilistic Model (DDPM) that
serves as a robot control policy. Instead of the usual generative modeling task
(generating images), this diffusion model generates ACTION SEQUENCES conditioned
on the robot's current observations.

Package structure:
    model/      — Neural network architectures and diffusion logic
    data/       — Dataset loading, normalization, and windowing
    env/        — PushT simulation environment wrappers

The key insight of Diffusion Policy (Chi et al., RSS 2023):
    Traditional behavioral cloning: obs → single action (or action chunk)
    Diffusion Policy:              obs + noise → iteratively denoised action chunk

By framing action prediction as iterative denoising, the model can represent
MULTIMODAL action distributions — multiple valid behaviors for the same
observation — which is impossible with a standard MSE-trained network that
always predicts the mean.
"""
