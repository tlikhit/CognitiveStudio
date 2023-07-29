# FreeNeRF

### Improving Few-shot Neural Rendering with Free Frequency Regularization

```{button-link} https://github.com/Jiawei-Yang/FreeNeRF
:color: primary
:outline:
Code
```

```{button-link} https://arxiv.org/abs/2303.07418
:color: primary
:outline:
Paper
```

## Running the Model

```bash
ns-train freenerf  # ...
```

## Method

FreeNeRF extends RegNeRF and adds two regularization methods.

### Frequency Regularization

The authors observe that a lower frequency positional encoding produces better
3D consistency, at the expense of blurred details.

The frequency regularization method is to mask higher frequencies of the
positional encoding, and gradually unmask them throughout training.

This causes the model to learn a consistent representation, then fine-tune the
details while maintaining the consistency.

### Occlusion Loss

The Occlusion Loss penalizes high density values near camera origins
("floaters"). While the floaters reconstruct training views well, they obscure
novel views.

## Uses

Like RegNeRF, FreeNeRF is good for sparse scenes. Additional regularization
techniques improve 3D consistency.
