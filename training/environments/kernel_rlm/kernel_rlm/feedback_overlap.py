"""Conservative eligibility for measured intermediate CUDA checks."""

import hashlib
import re

from kernel_rlm.submission_contract import validate_submission_source

RESERVE_BYTES = 2 * 1024**3
PROFILE_LIMITS = {
    "2f06c964cd1a3fcfbc3f6177ccfeab66b20187e0cf1c323c99bf95ec22f0819e": 512 * 1024**2,
    "abd43e11f40258b2113ef8721650dd787a76c8fbf01daded9405f251f5ade358": 4 * 1024**3,
}


def overlap_budget(reference: str, submission: str) -> tuple[int | None, str]:
    digest = hashlib.sha256(reference.strip().encode()).hexdigest()
    limit = PROFILE_LIMITS.get(digest)
    if limit is None:
        return None, "unprofiled_reference"
    valid, _reason = validate_submission_source(submission)
    if not valid:
        return None, "unsupported_submission"
    if re.search(
        r"cudaMalloc|cudaFree|cuMem|set_per_process_memory_fraction|"
        r"change_current_allocator|\bctypes\b|\bcupy\b|\bsubprocess\b",
        submission,
    ):
        return None, "external_allocation_or_allocator_control"
    return limit, "qualified_reference"
