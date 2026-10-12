"""
constants.py
============
Constants specific to the ESMFold inference application.

Not placed in the shared pmpnn constants because ESMFold is an
independent model (facebook/esmfold_v1) with no overlap with the ProteinMPNN
hyper-parameters.

Categories
----------
1. Model identity     — HuggingFace model ID
2. Multi-chain linker — glycine linker used to join chains for multi-chain input
3. Positional encoding — gap inserted between chains in position-ID space
4. Inference          — trunk chunk-size default
"""

# =============================================================================
# 1. Model identity
# =============================================================================
ESMFOLD_MODEL_ID: str = "facebook/esmfold_v1"

# =============================================================================
# 2. Multi-chain linker
# =============================================================================
# Glycine linker inserted between chains so the concatenated sequence can be
# fed to the single-chain ESMFold tokeniser.  25 G residues matches the
# recommended practice for virtual multi-chain folding with ESMFold.
LINKER: str = "G" * 25

# =============================================================================
# 3. Positional encoding
# =============================================================================
# Extra positional-index gap added *after* the linker when building per-chain
# position IDs.  This prevents the model from treating the linker residues as
# physically adjacent to the next chain.
CHAIN_POSITION_GAP: int = 200

# =============================================================================
# 4. Inference
# =============================================================================
# Default trunk chunk size.  Lowering this value reduces peak VRAM at the cost
# of slightly more compute.
DEFAULT_CHUNK_SIZE: int = 64
