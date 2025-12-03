from .diffusion import *

# The LayerDAG models depend on DGL's sparse C++ backend, which is not
# available on all platforms (e.g., macOS wheels may lack libdgl_sparse).
# Import them lazily and tolerate failures so users can still rely on
# diffusion-only functionality (used by the encoder) without requiring DGL.
try:  # pragma: no cover - platform‑dependent optional import
    from .layer_dag import *
except Exception:
    # Silently skip LayerDAG if its dependencies (e.g., DGL sparse library)
    # are unavailable. Code that needs LayerDAG should import it explicitly
    # and handle ImportError if necessary.
    pass
