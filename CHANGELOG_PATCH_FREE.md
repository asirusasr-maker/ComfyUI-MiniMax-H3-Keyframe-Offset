# Patch-free maintenance build

- Removed the import-time monkey patch of `comfy.ldm.minimax.model.PackedLayout.__init__`.
- Kept first/last frame offset behavior and pass `resolved_frame_index` via native `minimax_keyframes` conditioning.
- Removed the non-native `minimax_frame_count` conditioning key; the AV latent defines the target duration.
- Replaced the direct `torchaudio` resampler dependency with ComfyUI's `comfy.audio.resample`, matching current upstream MiniMax H3 implementation.
- Kept the AIO audio node's sampling, reference conditioning, Audio VAE decode, and `AUDIO` output flow.

No global ComfyUI class, method, function, or setting is modified at import time.

## Compatibility

Requires a ComfyUI build whose native MiniMax H3 `PackedLayout` accepts arbitrary `resolved_frame_index` values in `minimax_keyframes`. The current upstream implementation does. Older builds that only accept frame-zero / final-frame keyframes are not supported by this patch-free release; update ComfyUI rather than reintroducing a global monkey patch.

## Validation

Python syntax compilation and ZIP integrity checks passed. Full node execution was not run here because this environment does not contain the user's ComfyUI runtime, model weights, or GPU.
