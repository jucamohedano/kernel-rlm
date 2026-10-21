# Offline kernel test references

`kernel_softplus_reference.txt` preserves the exact reference source for task `lvl1-seed42-0011` from our seed42 materialization of [BytedTsinghua-SIA/CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K). Credit the original dataset and its authors; the upstream dataset card declares CC-BY-4.0. This reference text is dataset material, not newly authored framework code.

No code, shape, or constructor transformation was made. The fixture is text rather than an importable module so formatting tools cannot change the source identity. Its stripped UTF-8 SHA256 is `abd43e11f40258b2113ef8721650dd787a76c8fbf01daded9405f251f5ade358`, matching the existing measured-reference eligibility entry in `kernel_rlm.feedback_overlap`.

CPU tests hash and inspect this source; they do not execute it, allocate its tensors, run CUDA, or download a dataset. Keeping this small reference in the test suite avoids depending on a developer's private parent-workspace `notes/` directory. It does not broaden the runtime overlap allowlist.
