"""ComfyUI MiniMax H3 Keyframe Offset Node
Allows placing first/last keyframes at arbitrary frame indices,
giving the model more freedom to generate motion between them.
Also includes all-in-one audio generation node.
"""
import inspect
import logging
import comfy.ldm.minimax.model as minimax_model
from .keyframe_offset_node import MiniMaxH3KeyframeOffset, MiniMaxH3AudioGenerator


def _patch_minimax_keyframes():
    try:
        original_init = minimax_model.PackedLayout.__init__
        params = inspect.signature(original_init).parameters
        accepts_frame_count = "frame_count" in params

        # 0.34+: ядро само ставит любой resolved_frame_index через
        # cursor + FRAME_RESCALE * index. Патч не нужен.
        if not accepts_frame_count:
            logging.info(
                "MiniMax H3 Keyframe Offset: PackedLayout has no frame_count "
                "(ComfyUI 0.34+) — stock layout already supports arbitrary "
                "keyframe indices; patch skipped"
            )
            return

        # Старый core: first/last only + frame_count. Обходим ограничение.
        def patched_init(
            self,
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=None,
            refs=None,
            frame_count=None,
        ):
            original_indices = []
            if keyframes:
                for kf in keyframes:
                    original_indices.append(kf.get("resolved_frame_index", 0))
                    kf["resolved_frame_index"] = 0

            kwargs = {"keyframes": keyframes, "refs": refs}
            if accepts_frame_count:
                kwargs["frame_count"] = frame_count
            original_init(
                self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs
            )

            if keyframes and original_indices:
                cond_segments = [
                    (a, b) for a, b, kind in self.segments if kind == "cond"
                ]
                # origin целевого timeline (после text + refs)
                origin = float(text_len)
                for blk in refs or ():
                    try:
                        origin += float(minimax_model._ref_t_span(blk))
                    except Exception:
                        break

                for i, (a, b) in enumerate(cond_segments):
                    if i < len(original_indices):
                        real_idx = original_indices[i]
                        cond_t = origin + minimax_model.FRAME_RESCALE * float(real_idx)
                        self.position_ids[a:b, 0] = cond_t

                for i, kf in enumerate(keyframes):
                    if i < len(original_indices):
                        kf["resolved_frame_index"] = original_indices[i]

        minimax_model.PackedLayout.__init__ = patched_init
        logging.info(
            "MiniMax H3 Keyframe Offset: legacy PackedLayout patch applied "
            "(frame_count present)"
        )
    except Exception as e:
        logging.warning(f"MiniMax H3 Keyframe Offset: Patch not applied ({e})")


_patch_minimax_keyframes()

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3KeyframeOffset": MiniMaxH3KeyframeOffset,
    "MiniMaxH3AudioGenerator": MiniMaxH3AudioGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3KeyframeOffset": "MiniMax H3 Keyframe Offset",
    "MiniMaxH3AudioGenerator": "MiniMax H3 Audio Generator (AIO)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]