# Adapted from City96/ComfyUI-GGUF for Lollms
# Stripped of 'comfy' dependencies
import gguf
import torch
import logging
from .dequant import dequantize_tensor, is_quantized

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self
    
    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This is responsible for de-quantizing on the fly
    """
    dequant_dtype = None
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return None
        # dequantize tensor 
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = bias.to(dtype=bias_dtype, device=device)

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = weight.to(dtype=dtype, device=device)
        return weight, bias

class GGMLLinear(GGMLLayer, torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        # We start with dummy weights that will be replaced by GGMLTensors
        
    def forward(self, input):
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)
        return super().forward(input)

class GGMLConv2d(GGMLLayer, torch.nn.Conv2d):
    def forward(self, input):
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
        return super().forward(input)

class GGMLEmbedding(GGMLLayer, torch.nn.Embedding):
    def forward(self, input):
        if self.is_ggml_quantized():
             weight, _ = self.cast_bias_weight(self, device=input.device)
             return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
            )
        return super().forward(input)

class GGMLLayerNorm(GGMLLayer, torch.nn.LayerNorm):
    def forward(self, input):
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        return super().forward(input)

class GGMLGroupNorm(GGMLLayer, torch.nn.GroupNorm):
    def forward(self, input):
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
        return super().forward(input)
