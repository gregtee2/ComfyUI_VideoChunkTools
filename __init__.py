# MIT License — Copyright (c) 2026 Greg Tee — see LICENSE file
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# WanChunkedI2VSampler — requires ComfyUI-WanVideoWrapper at runtime (not import time).
# If the import fails for any reason, the base VideoChunkTools nodes still work fine.
try:
    from .wan_chunked_sampler import (
        NODE_CLASS_MAPPINGS as _WAN_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _WAN_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_WAN_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_WAN_NAMES)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
