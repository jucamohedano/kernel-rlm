# Offline provenance validation example

This is a self-contained example for the provenance validator, not a launchable training experiment. It uses the existing checked-in vector-add task and public source/documentation paths. There are no private development notes, downloaded model/data dependencies, sampled trajectories, GPU allocations, or paid services.

From the repository root:

```bash
PYTHONPATH=training/environments/kernel_rlm python -c 'from pathlib import Path; from kernel_factory import validate_registry; print(validate_registry(Path.cwd(), Path("examples/factory_cpu")))'
```

This validates the registry fields, task identity, local file existence, and dataset hashes without executing the reference program. It does not attest historical training results or authorize a launch.

The historical dashboard registry contains research records that reference separately archived inputs and private notes. It is not a clean-clone executable example and is not rewritten to make missing historical provenance appear valid. CPU regression tests validate this checked-in example instead, while retaining fail-closed tests for missing provenance, unknown datasets, absent inputs, tampered hashes, and unsafe paths. Completing the historical launch registry's portability remains a separate task.
