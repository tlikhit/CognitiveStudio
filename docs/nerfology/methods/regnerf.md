# RegNeRF

### Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs

```{button-link} https://github.com/google-research/google-research/tree/master/regnerf
:color: primary
:outline:
Code
```

```{button-link} https://arxiv.org/abs/2112.00724.pdf
:color: primary
:outline:
Paper
```

## Running the Model

```bash
ns-train regnerf  # ...
```

## Method

RegNeRF regularizes training to model sparse (few input views) scenes better.
Three additional components are added to the training process.

### Depth Smoothness Loss

The Depth Smoothness Loss incentivizes smooth geometry. This makes it more
likely that the learned scene has "3D consistent" objects, creating better
reconstructions from new views.

A set of random camera poses is defined. In our implementation, we sample poses
a distance ``r`` from the origin and looking at the origin.

A small (8x8) depth map is rendered from a batch of these poses each training
step. The loss penalizes unsmooth depth maps by comparing the values of
neighboring pixels.

### Color Likelihood

The Color Likelihood Loss tries to improve color accuracy.

A separate RealNVP model is trained to predict how likely certain arrangements
of colors are. Training on "diverse natural images", the model learns the
distribution of how often each 8x8 patch appears.

This is used with the random poses described above (Depth Smoothness Loss). The
color output from the poses is passed through the model. Using negative log
likelihood, a loss is computed.

In both the official and this (CognitiveStudio) implementation, Color Likelihood
loss is omitted.

### Sample Space Annealing

Sample Space Annealing reduces the sampling space at the beginning of training.
This aims to create a "3D consistent" representation, like Depth Smoothness
Loss.

For each ray (which has a near and far plane), the middle is computed as the
average of the near and far.

At the beginning of training, the annealed nears and fars are computed: They
begin close to the mid point, and linearly move back to their original values.

This has the effect of gradually increasing the available scene sampling space,
causing the model to learn a coherent object in the beginning of training.

## Results

|  Num. GT images  |  Official PSNR  |  CognitiveStudio's PSNR  |
| ---------------- | --------------- | ------------------------ |
| 3                | 18.89           |                          |
| 6                | 22.20           |                          |
| 9                | 24.93           |                          |
