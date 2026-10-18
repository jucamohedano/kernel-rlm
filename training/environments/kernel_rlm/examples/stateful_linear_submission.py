import torch
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = """
#include <torch/extension.h>
torch::Tensor launch(torch::Tensor inputs, torch::Tensor weight, torch::Tensor bias);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void linear_kernel(
    const float* inputs, const float* weight, const float* bias,
    float* output, int rows, int input_width, int output_width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * output_width) return;
    int row = index / output_width;
    int column = index % output_width;
    float value = bias[column];
    for (int offset = 0; offset < input_width; ++offset) {
        value += inputs[row * input_width + offset] * weight[column * input_width + offset];
    }
    output[index] = value;
}

torch::Tensor launch(torch::Tensor inputs, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(inputs.is_cuda() && weight.is_cuda() && bias.is_cuda());
    TORCH_CHECK(inputs.is_contiguous() && weight.is_contiguous() && bias.is_contiguous());
    TORCH_CHECK(inputs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32);
    int rows = inputs.size(0);
    int input_width = inputs.size(1);
    int output_width = weight.size(0);
    auto output = torch::empty({rows, output_width}, inputs.options());
    int count = rows * output_width;
    if (count > 0) {
        linear_kernel<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            inputs.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
            output.data_ptr<float>(), rows, input_width, output_width);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
"""

module = load_inline(
    name="kernel_stateful_linear",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=["launch"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(torch.nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.linear = torch.nn.Linear(input_width, output_width)

    def forward(self, inputs):
        values = module.launch(inputs, self.linear.weight, self.linear.bias)
        return {"values": values, "shape": tuple(inputs.shape)}
