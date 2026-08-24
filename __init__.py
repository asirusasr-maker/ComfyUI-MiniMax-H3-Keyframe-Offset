"""ComfyUI MiniMax H3 Keyframe Offset Node
Allows placing first/last keyframes at arbitrary frame indices,
giving the model more freedom to generate motion between them.
Also includes all-in-one audio generation node.
"""
import logging
import comfy.ldm.minimax.model as minimax_model
from .keyframe_offset_node import MiniMaxH3KeyframeOffset, MiniMaxH3AudioGenerator

# ==============================================================================
# ПАТЧ ДЛЯ ОБХОДА ОГРАНИЧЕНИЯ ЯДРА COMFYUI
# ==============================================================================
def _patch_minimax_keyframes():
    try:
        original_init = minimax_model.PackedLayout.__init__
        
        def patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
            original_indices = []
            
            # Сохраняем оригинальные индексы и временно ставим 0 для прохождения проверки
            if keyframes:
                for kf in keyframes:
                    original_indices.append(kf.get("resolved_frame_index", 0))
                    kf["resolved_frame_index"] = 0 
            
            # Вызываем оригинальный __init__
            original_init(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes, refs, frame_count)
            
            # Восстанавливаем реальные индексы и пересчитываем position_ids
            if keyframes and original_indices:
                cond_segments = [(a, b) for a, b, kind in self.segments if kind == "cond"]
                
                for i, (a, b) in enumerate(cond_segments):
                    if i < len(original_indices):
                        real_idx = original_indices[i]
                        cond_t = float(text_len) + minimax_model.FRAME_RESCALE * float(real_idx)
                        self.position_ids[a:b, 0] = cond_t
                        
                # Восстанавливаем оригинальные значения в словарях
                for i, kf in enumerate(keyframes):
                    if i < len(original_indices):
                        kf["resolved_frame_index"] = original_indices[i]

        minimax_model.PackedLayout.__init__ = patched_init
        logging.info("MiniMax H3 Keyframe Offset: Patch applied successfully")
    except Exception as e:
        logging.warning(f"MiniMax H3 Keyframe Offset: Patch not applied ({e})")

# Применяем патч при загрузке
_patch_minimax_keyframes()
# ==============================================================================

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
