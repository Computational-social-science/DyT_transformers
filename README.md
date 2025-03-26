# Evalution: Transformer (Layer Norm) vs Transformer (DyT)

## DyT Overview
Meta proposes Dynamic Tanh (DyT), an element-wise operation defined as: DyT(x) = tanh(αx), where α is a learnable scaler. DyT is designed to replace normalization layers in Transformers. Models with DyT achieves similar or better performance than their normalized counterparts.
<img src="https://github.com/Computational-social-science/DyT_transformers/blob/main/before_after.svg" />

**Figure 1**. Left: original Transformer block. Right: block with our proposed Dynamic Tanh (DyT) layer. DyT is a straightforward replacement for commonly used Layer Norm (Ba et al., 2016) (in some cases RMSNorm (Zhang and Sennrich, 2019)) layers. Transformers with DyT match or exceed the performance of their normalized counterparts.

**Notes**: The figure is retrieved from the DyT (https://github.com/jiachenzhu/DyT) library.

## Implementation
Meta's DyT module can be implemented in 9 lines of PyTorch code:

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

## Comprehensive Evalution
For communications, here we present a comprehensive framework for evaluting the pros and cons of Dynamic Tanh (DyT) aginst the vanilla Transformers:
<img src="https://github.com/Computational-social-science/DyT_transformers/blob/main/transformer_comparison.svg" />

## Acknowledgement
This repository is built using the DyT (https://github.com/jiachenzhu/DyT) library.

## License
This project is released under the MIT license. Please see the LICENSE file for more information.
