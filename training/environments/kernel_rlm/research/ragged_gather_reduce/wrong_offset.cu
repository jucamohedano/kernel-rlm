// Build instead of (not alongside) control.cu. Use the mask-offset fixture:
// all its gather IDs are legal, even under the wrong mask.
#define RAGGED_NEGATIVE_CONTROL
#include "control.cu"
