"""ComfyUI MiniMax H3 custom nodes.

This package intentionally does not monkey-patch ComfyUI internals.
Keyframe offsets are passed through the native ``minimax_keyframes``
conditioning field supported by current ComfyUI MiniMax H3.
"""

from .keyframe_offset_node import MiniMaxH3KeyframeOffset, MiniMaxH3AudioGenerator

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3KeyframeOffset": MiniMaxH3KeyframeOffset,
    "MiniMaxH3AudioGenerator": MiniMaxH3AudioGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3KeyframeOffset": "MiniMax H3 Keyframe Offset",
    "MiniMaxH3AudioGenerator": "MiniMax H3 Voice Studio (AIO)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
