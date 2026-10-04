import torch
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor left, torch::Tensor right);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_add", &vector_add_cuda);
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

__global__ void vector_add_kernel(
    const float* left,
    const float* right,
    float* output,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor left, torch::Tensor right) {
    TORCH_CHECK(left.is_cuda() && right.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(left.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(left.sizes() == right.sizes(), "inputs must have equal shapes");
    auto output = torch::empty_like(left);
    const int threads = 256;
    const int blocks = (left.numel() + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        left.data_ptr<float>(),
        right.data_ptr<float>(),
        output.data_ptr<float>(),
        left.numel()
    );
    return output;
}
"""


_module = load_inline(
    name="kernel_rlm_vector_add",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def cuda_forward(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _module.vector_add(left, right)
