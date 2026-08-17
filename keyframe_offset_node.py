import math
import torch
import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import comfy.sample
import comfy.samplers
import node_helpers

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_SAMPLE_RATE = 32000

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n

def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)

def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)

def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _to_audio_dict(decoded, sample_rate=AUDIO_SAMPLE_RATE):
    """Normalize VAE decode output to ComfyUI AUDIO format: {"waveform": [B,C,L], "sample_rate": int}."""
    if isinstance(decoded, dict):
        waveform = decoded.get("waveform", decoded)
        sample_rate = int(decoded.get("sample_rate", sample_rate))
    else:
        waveform = decoded

    if not torch.is_tensor(waveform):
        raise TypeError(f"Audio VAE decode returned unsupported type: {type(waveform)!r}")

    if waveform.ndim == 2:
        if waveform.shape[-1] <= 8 and waveform.shape[0] > waveform.shape[-1]:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        if waveform.shape[-1] <= 8 and waveform.shape[1] > waveform.shape[-1]:
            waveform = waveform.movedim(-1, 1)
    else:
        raise ValueError(f"Audio VAE decode must be [B,C,L] or [B,L,C], got {tuple(waveform.shape)}")

    return {
        "waveform": waveform.float().contiguous().cpu(),
        "sample_rate": int(sample_rate),
    }


class MiniMaxH3KeyframeOffset:
    """MiniMax H3 Image to Video with arbitrary keyframe frame offsets."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range ~124-362)"}),
                "first_frame_offset": ("INT", {"default": 0, "min": 0, "max": 3599, "step": 1,
                    "tooltip": "Frame index where first_frame is injected. 0 = very first frame. Clamped to [0, frame_count-1]."}),
                "last_frame_offset": ("INT", {"default": -1, "min": -1, "max": 3599, "step": 1,
                    "tooltip": "Frame index where last_frame is injected. -1 = last frame. Clamped to [0, frame_count-1]."}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax"

    def execute(self, clip, vae, prompt, width, height, length,
                first_frame_offset=0, last_frame_offset=-1,
                first_frame=None, last_frame=None):
        latent, frame_count = _empty_av_latent(width, height, length)

        if last_frame_offset < 0:
            last_frame_offset = frame_count - 1

        original_first = first_frame_offset
        original_last = last_frame_offset

        first_frame_offset = max(0, min(first_frame_offset, frame_count - 1))
        last_frame_offset = max(0, min(last_frame_offset, frame_count - 1))

        if original_first != first_frame_offset or original_last != last_frame_offset:
            print(
                f"[MiniMax H3 Keyframe Offset] Offsets clamped to [0, {frame_count - 1}]: "
                f"first_frame_offset {original_first}->{first_frame_offset}, "
                f"last_frame_offset {original_last}->{last_frame_offset}"
            )

        images = []
        keyframes = []
        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": first_frame_offset, "image": img})
        if last_frame is not None:
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": last_frame_offset, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })

        return (cond, latent)


class MiniMaxH3AudioGenerator:
    """All-in-one MiniMax H3 Audio Generator.

    Uses sample_custom. CFG hardcoded to 1.0 (native H3 is guidance-distilled).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "bird singing, forest ambience",
                    "tooltip": "Describe the sound you want to generate"
                }),
                "length": ("INT", {
                    "default": 124,
                    "min": 5,
                    "max": 3600,
                    "step": 17,
                    "tooltip": "Audio duration in frames. 124 = ~5 seconds at 24fps"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of sampling steps"
                }),
                "sampler_name": (["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "res_multistep"], {"default": "res_multistep"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"], {"default": "simple"}),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoise strength (1.0 = full generation)"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax"

    def execute(self, model, clip, vae, audio_vae, prompt, length, seed, steps, sampler_name, scheduler, denoise=1.0):
        cfg = 1.0  # native MiniMax H3 is guidance-distilled

        width, height = 320, 320
        latent, frame_count = _empty_av_latent(width, height, length)

        tokens = clip.tokenize(prompt, images=[])
        positive = clip.encode_from_tokens_scheduled(tokens)
        positive = node_helpers.conditioning_set_values(positive, {"minimax_frame_count": frame_count})

        negative = []

        sampler = comfy.samplers.sampler_object(sampler_name)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

        if denoise < 1.0:
            if denoise <= 0.0:
                audio_latent = latent["samples"].tensors[1]
                decoded = audio_vae.decode(audio_latent)
                return (_to_audio_dict(decoded),)
            total_steps = int(steps * denoise)
            sigmas = sigmas[len(sigmas) - total_steps - 1:]

        seed = int(seed) & 0xffffffffffffffff
        noise = comfy.sample.prepare_noise(latent["samples"], seed)

        samples = comfy.sample.sample_custom(
            model=model,
            noise=noise,
            cfg=cfg,
            sampler=sampler,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            latent_image=latent["samples"],
            noise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed
        )

        if hasattr(samples, "tensors"):
            audio_latent = samples.tensors[1]
        elif isinstance(samples, dict):
            audio_latent = samples["samples"].tensors[1]
        else:
            audio_latent = samples

        decoded = audio_vae.decode(audio_latent)
        audio = _to_audio_dict(decoded)

        return (audio,)