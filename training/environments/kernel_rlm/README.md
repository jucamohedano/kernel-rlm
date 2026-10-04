# kernel_rlm

CUDA kernel tasks for the local RLM training loop.

The environment keeps Python as the persistent REPL language. A submission is Python source
that defines CPP_SOURCE, CUDA_SOURCE, and cuda_forward(*tensors). The submission compiles
the CUDA implementation through torch.utils.cpp_extension and calls the resulting extension.

The verifier imports the submission, compiles the extension, checks correctness on fresh CUDA
inputs, and benchmarks only after correctness passes. Verification runs in a separate process.

The tiny vector-add task is in kernel_rlm/tasks/vector_add.jsonl. A known-good submission is
in examples/vector_add_submission.py.

The package is intentionally compatible with the existing rlm_train.RLMTrainEnv. The generic
RLM loop and the historical triton_rlm environment remain separate.
