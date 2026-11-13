import re
import json
import math
import torch
import comfy.audio
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
REF_IMAGE_SHORT_EDGE = 2048
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

def _encode_ref_audio(audio_vae, audio):
    """Native MiniMax H3 ref-audio encode. Returns (latent [1,32,2,T], T)."""
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = comfy.audio.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]

def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _to_audio_dict(decoded, sample_rate=AUDIO_SAMPLE_RATE):
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


def _build_ref2va_refs(vae, audio_vae, width, height, ref_image_size, ref_image, ref_audio):
    """Build native-style ref_items (tokenizer) + ref_blocks (DiT minimax_refs).

    Order matches MiniMaxH3ReferenceToVideo:
      images first -> standalone audio last.
    Ordinals are 1-based: <Picture 1>, <Audio 1>.
    """
    ref_items = []
    ref_blocks = []

    if ref_image is not None:
        img = ref_image[:1]
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / max(1, w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / max(1, min(w, h)))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img, tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": th // 16,
            "latent_w": tw // 16,
            "latent": z,
        })

    if ref_audio is not None:
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, ref_audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
        })

    return ref_items, ref_blocks


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
            # Native ComfyUI MiniMax H3 reads resolved_frame_index directly
            # from minimax_keyframes. No global PackedLayout patch is needed.
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
            })

        return (cond, latent)


def _build_speech_instruction(text, voice_type="Female", custom_voice_description="",
                              voice_timbre="Natural", speaking_style="Natural Conversation",
                              emotion="Neutral", emotion_intensity=50,
                              speech_speed=1.0, speech_pitch=0.0):
    """Build plain-text speech instructions for the current H3 text-conditioning path.

    These are prompt-level hints, not guaranteed acoustic controls. Keep the supplied
    script verbatim and avoid adding model-specific special tokens that are not verified.
    """
    script = str(text or "").strip()
    if not script:
        raise ValueError("Speech Generation requires non-empty speech_text (or prompt fallback).")

    voice = str(voice_type or "Female")
    if voice == "Custom":
        custom = str(custom_voice_description or "").strip()
        voice_description = custom if custom else "a natural, clearly articulated voice"
    else:
        voice_description = voice.lower() + " voice"

    timbre = str(voice_timbre or "Natural")
    style = str(speaking_style or "Natural Conversation")
    mood = str(emotion or "Neutral")
    intensity = max(0, min(100, int(emotion_intensity)))
    speed = max(0.5, min(2.0, float(speech_speed)))
    pitch = max(-12.0, min(12.0, float(speech_pitch)))

    if speed < 0.9:
        speed_hint = "slightly slower than normal"
    elif speed > 1.1:
        speed_hint = "slightly faster than normal"
    else:
        speed_hint = "a natural conversational pace"

    if pitch <= -2.0:
        pitch_hint = "somewhat lower-pitched"
    elif pitch >= 2.0:
        pitch_hint = "somewhat higher-pitched"
    else:
        pitch_hint = "natural pitch"

    intensity_hint = (
        "subtle emotional expression" if intensity < 30 else
        "moderate emotional expression" if intensity < 70 else
        "strong emotional expression"
    )

    instruction = (
        "Generate spoken voice audio. Speak the exact script below; do not paraphrase, "
        "translate, summarize, omit, or add words.\n"
        f"Voice description: {voice_description}.\n"
        f"Vocal timbre: {timbre.lower()}.\n"
        f"Speaking style: {style.lower()}.\n"
        f"Emotion: {mood.lower()} with {intensity_hint}.\n"
        f"Delivery pace: {speed_hint}.\n"
        f"Pitch direction: {pitch_hint}.\n"
        "Use clear pronunciation, natural phrasing, and appropriate pauses. "
        "Do not add background music or sound effects unless explicitly requested.\n\n"
        "SCRIPT:\n"
        f"{script}"
    )
    return instruction



def _parse_expressive_script(script):
    """Parse simple inline stage directions into (instruction, spoken_text) segments.

    Supported syntax:
      [happy] Hello there.
      [whispering, sad] I missed you.
      [pause=700ms]
      [laughing]
    Tags are kept as prompt-level directions. The parser never sends bracket tags as
    spoken words, and does not claim deterministic acoustic control.
    """
    text = str(script or "").strip()
    if not text:
        raise ValueError("Expressive TTS requires non-empty speech_text.")

    token_re = re.compile(r"(\[[^\]]+\])")
    parts = token_re.split(text)
    segments = []
    pending_tags = []

    for part in parts:
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            tag = part[1:-1].strip()
            if tag.lower().startswith("pause="):
                raw = tag.split("=", 1)[1].strip().lower()
                match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds)?", raw)
                if not match:
                    raise ValueError(f"Invalid pause tag [{tag}]. Use e.g. [pause=500ms] or [pause=1.2s].")
                amount = float(match.group(1))
                unit = match.group(2) or "ms"
                milliseconds = amount * (1000 if unit in ("s", "sec", "seconds") else 1)
                milliseconds = int(max(0, min(10000, milliseconds)))
                segments.append({"kind": "pause", "duration_ms": milliseconds})
            else:
                pending_tags.append(tag)
        else:
            spoken = part.strip()
            if spoken:
                segments.append({
                    "kind": "speech",
                    "directions": pending_tags[:],
                    "text": spoken,
                })
                pending_tags.clear()

    # A tag-only script is not useful and should fail early rather than synthesize silence.
    if not any(s["kind"] == "speech" for s in segments):
        raise ValueError("Expressive TTS script contains no spoken text.")
    return segments


def _build_expressive_instruction(script, voice_type="Female",
                                  custom_voice_description="",
                                  voice_timbre="Natural",
                                  default_style="Natural Conversation",
                                  default_emotion="Neutral", default_intensity=50,
                                  default_speed=1.0, default_pitch=0.0,
                                  voice_reference=False):
    """Build expressive instructions with a distinct voice-identity layer.

    If a reference audio is connected, it is the primary identity source. Without a
    reference, voice_type/custom_voice_description/voice_timbre define the requested
    voice identity. Performance controls remain independent in both cases.
    """
    segments = _parse_expressive_script(script)

    voice_type = str(voice_type or "Female")
    custom = str(custom_voice_description or "").strip()
    timbre = str(voice_timbre or "Natural")
    style = str(default_style or "Natural Conversation")
    mood = str(default_emotion or "Neutral")
    intensity = max(0, min(100, int(default_intensity)))
    speed = max(0.5, min(2.0, float(default_speed)))
    pitch = max(-12.0, min(12.0, float(default_pitch)))

    if voice_type == "Custom":
        identity_description = custom or "a natural, clearly articulated custom voice"
    else:
        identity_description = f"{voice_type.lower()} voice"

    timbre_descriptions = {
        "Natural": "natural, balanced vocal timbre",
        "Deep": "deep, full-bodied resonance",
        "Soft": "soft, gentle vocal texture",
        "Bright": "bright, clear vocal tone",
        "Warm": "warm, rounded vocal tone",
        "Breathy": "lightly breathy vocal texture",
        "Resonant": "strong, resonant vocal presence",
        "Character": "distinctive character-style vocal texture",
    }
    timbre_description = timbre_descriptions.get(timbre, timbre.lower())

    intensity_hint = (
        "subtle" if intensity < 30 else
        "moderate" if intensity < 70 else
        "strong"
    )
    if speed < 0.9:
        speed_hint = "slower than a normal conversational pace"
    elif speed > 1.1:
        speed_hint = "faster than a normal conversational pace"
    else:
        speed_hint = "a natural conversational pace"
    if pitch <= -2.0:
        pitch_hint = "somewhat lower-pitched"
    elif pitch >= 2.0:
        pitch_hint = "somewhat higher-pitched"
    else:
        pitch_hint = "natural pitch"

    lines = [
        "Generate one continuous expressive spoken-voice performance.",
        "Speak only the script text shown in the performance segments. Do not speak bracketed directions.",
        "Preserve the exact wording, language, punctuation, and order. Do not paraphrase, translate, omit, or add words.",
        "",
        "VOICE IDENTITY:",
    ]
    if voice_reference:
        lines.extend([
            "AUDIO REFERENCE: <Audio 1>.",
            "The supplied audio reference is the PRIMARY source of speaker identity.",
            "Prioritize the reference speaker's recognizable identity, perceived age, vocal gender presentation, accent, timbre, resonance, articulation, and natural speaking manner.",
            "Do not replace the reference identity with a conflicting generic voice type. Any voice-type selection is secondary and must not override the reference.",
            f"Requested vocal texture: {timbre_description}; apply only where compatible with the reference identity.",
        ])
    else:
        lines.extend([
            f"Requested voice identity: {identity_description}.",
            f"Vocal timbre: {timbre_description}.",
            "Maintain this requested voice identity consistently throughout the performance.",
        ])
    lines.extend([
        "",
        "EXPRESSIVE PERFORMANCE:",
        f"Overall speaking style: {style.lower()}.",
        f"Default emotion: {mood.lower()} with {intensity_hint} expression (intensity {intensity}/100).",
        f"Delivery pace: {speed_hint} (requested speed {speed:.2f}x).",
        f"Pitch direction: {pitch_hint} (requested pitch direction {pitch:.1f}).",
        "Inline delivery directions override the default emotion and style locally where appropriate.",
        "",
        "PERFORMANCE SCRIPT:",
    ])
    for seg in segments:
        if seg["kind"] == "pause":
            lines.append(f"[PAUSE for {seg['duration_ms']} milliseconds]")
        else:
            directions = ", ".join(seg["directions"]) if seg["directions"] else "use the default delivery"
            lines.append(f"[DELIVERY: {directions}]")
            lines.append(seg["text"])
    lines.extend([
        "",
        "Treat laughter, crying, sighing, whispering, shouting, and emotional changes as performance directions, not literal words.",
        "Make transitions natural. Maintain consistent speaker identity throughout.",
        "These directions are prompt guidance, not guaranteed frame-accurate or independently controllable acoustic events.",
    ])
    return "\n".join(lines), segments



def _build_voice_cloning_instruction(text, voice_reference_mode="Clone voice identity",
                                     speaking_style="Natural Conversation",
                                     emotion="Neutral", emotion_intensity=50,
                                     speech_speed=1.0, additional_instructions=""):
    """Build a reference-conditioned speech prompt using the native <Audio 1> marker.

    The audio reference is supplied separately through native MiniMax H3 ref conditioning.
    This prompt does not claim guaranteed voice cloning; it asks the model to use the
    reference as a voice/style guide while speaking the exact new script.
    """
    script = str(text or "").strip()
    if not script:
        raise ValueError("Voice Cloning TTS requires non-empty speech_text.")

    ref_mode = str(voice_reference_mode or "Clone voice identity")
    style = str(speaking_style or "Natural Conversation")
    mood = str(emotion or "Neutral")
    intensity = max(0, min(100, int(emotion_intensity)))
    speed = max(0.5, min(2.0, float(speech_speed)))

    intensity_hint = (
        "subtle" if intensity < 30 else
        "moderate" if intensity < 70 else
        "strong"
    )
    pace_hint = (
        "slower than the reference's usual pace" if speed < 0.9 else
        "faster than the reference's usual pace" if speed > 1.1 else
        "a natural pace consistent with the reference"
    )

    extra = str(additional_instructions or "").strip()
    extra_block = f"\nAdditional delivery direction: {extra}" if extra else ""

    return (
        "Generate a spoken voice performance using the supplied audio reference.\n"
        "<Audio 1> is the voice reference. Use it as the primary guide for the speaker's "
        "vocal identity, accent, timbre, resonance, articulation, and natural speaking manner.\n"
        f"Reference handling: {ref_mode}.\n"
        "Do not reproduce the reference recording's words or background sounds. "
        "Speak the new script below instead. Preserve the recognizable vocal qualities "
        "of <Audio 1> as closely as the model allows.\n"
        f"Speaking style: {style.lower()}.\n"
        f"Emotion: {mood.lower()} with {intensity_hint} expression.\n"
        f"Delivery pace: {pace_hint}.\n"
        "Speak clearly and naturally. Do not paraphrase, translate, summarize, omit, "
        "or add words. Do not add background music or sound effects."
        f"{extra_block}\n\n"
        "NEW SCRIPT:\n"
        f"{script}"
    )



class MiniMaxH3AudioGenerator:
    """All-in-one MiniMax H3 Audio Generator with optional Ref2VA-style refs.

    When use_references is on, builds native minimax_ref_items / minimax_refs
    (same path as MiniMax H3 Reference to Video), simplified to:
      - one ref image  -> <Picture 1>
      - one ref audio  -> <Audio 1>

    Outputs AUDIO first and the effective conditioning prompt second. CFG is hardcoded to 1.0.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "generation_mode": (["Audio Generation", "Expressive TTS"], {
                    "default": "Audio Generation",
                    "tooltip": "Two top-level modes: general audio generation or all speech workflows inside Expressive TTS."
                }),
                "expressive_tts_mode": (["Single Voice", "Dual Voice Dialogue"], {
                    "default": "Single Voice",
                    "tooltip": "Single Voice uses speech_text and optional ref_audio. Dual Voice Dialogue uses a script and two independent audio references."
                }),
                "speech_language": (["Auto", "English", "Russian", "Chinese", "Japanese", "Korean", "Spanish", "French", "German", "Portuguese", "Italian", "Custom"], {
                    "default": "Auto",
                    "tooltip": "Default language for Expressive TTS. Dual Voice Dialogue lines can override it with [lang=English], [lang=Russian], etc."
                }),
                "custom_speech_language": ("STRING", {"default": "", "multiline": False, "tooltip": "Used only when Speech Language is Custom."}),
                "dialogue_script": ("STRING", {
                    "multiline": True,
                    "default": "[Speaker 1]\nHello! How are you?\n\n[Speaker 2]\nI'm doing well, thank you!",
                    "tooltip": "Use [Speaker 1] and [Speaker 2] before each line. Optional [happy], [sad], [whispering] and [pause=700ms] tags."
                }),
                "speaking_speed_hint": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Approximate speaking pace and duration-estimation hint; not precise acoustic timing control."
                }),
                "duration_profile": (["Compact", "Balanced", "Expressive"], {
                    "default": "Balanced",
                    "tooltip": "Adaptive speech-duration budget for Dual Voice Dialogue."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamic_prompts": True,
                    "default": "bird singing, forest ambience",
                    "tooltip": "Audio Generation prompt. In Speech Generation this is used only when speech_text is empty."
                }),
                "speech_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Exact text to speak in Speech Generation. This field must not be empty."
                }),
                "voice_type": (["Female", "Male", "Child", "Teenager", "Elderly Female", "Elderly Male", "Androgynous", "Custom"], {
                    "default": "Female",
                    "tooltip": "Expressive TTS Voice Identity: used when no ref_audio is connected. With ref_audio, the reference identity takes priority."
                }),
                "custom_voice_description": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom Voice Identity description. Used when Voice Type is Custom and no ref_audio is connected."
                }),
                "voice_timbre": (["Natural", "Deep", "Soft", "Bright", "Warm", "Breathy", "Resonant", "Character"], {
                    "default": "Natural"
                }),
                "speaking_style": (["Natural Conversation", "Narration", "Commercial", "Podcast", "Whispering", "Storytelling", "Expressive Acting"], {
                    "default": "Natural Conversation"
                }),
                "emotion": (["Neutral", "Happy", "Sad", "Romantic", "Angry", "Confident", "Calm", "Excited"], {
                    "default": "Neutral"
                }),
                "emotion_intensity": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Prompt hint only; not a calibrated acoustic intensity control."
                }),
                "speech_speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Prompt-level pace hint. Actual timing may vary."
                }),
                "speech_pitch": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Prompt-level pitch direction hint, not guaranteed semitone control."
                }),
                "voice_reference_mode": (["Clone voice identity", "Match voice style"], {
                    "default": "Clone voice identity",
                    "tooltip": "Instruction applied to the supplied <Audio 1> reference. Actual identity preservation depends on the checkpoint."
                }),
                "cloning_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional additional delivery instructions for the cloned/reference voice."
                }),
                "expressive_default_style": (["Natural Conversation", "Narration", "Commercial", "Podcast", "Storytelling", "Acting"], {
                    "default": "Natural Conversation"
                }),
                "expressive_default_emotion": (["Neutral", "Happy", "Sad", "Romantic", "Angry", "Confident", "Calm", "Excited"], {
                    "default": "Neutral"
                }),
                "expressive_default_intensity": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 1
                }),
                "expressive_default_speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05
                }),
                "expressive_default_pitch": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5
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
                "use_references": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Ref2VA-style <Picture 1> / <Audio 1> conditioning (native minimax_refs path)."
                }),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "Reference image sizing (same as native Reference to Video). 'max' = 2048 short edge, slower but better identity."
                }),
            },
            "optional": {
                "ref_image": ("IMAGE", {
                    "tooltip": "Reference image -> <Picture 1> in the prompt"
                }),
                "ref_audio": ("AUDIO", {
                    "tooltip": "Single Voice reference -> <Audio 1>. In Expressive TTS this is the sole voice identity reference."
                }),
                "ref_audio_1": ("AUDIO", {"tooltip": "Dual Voice Dialogue identity reference for Speaker 1."}),
                "ref_audio_2": ("AUDIO", {"tooltip": "Dual Voice Dialogue identity reference for Speaker 2."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "conditioning_prompt")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax"

    def execute(self, model, clip, vae, audio_vae, generation_mode, prompt, speech_text,
                voice_type, custom_voice_description, voice_timbre, speaking_style,
                emotion, emotion_intensity, speech_speed, speech_pitch,
                voice_reference_mode, cloning_instructions,
                expressive_default_style, expressive_default_emotion,
                expressive_default_intensity, expressive_default_speed, expressive_default_pitch,
                length, seed, steps, sampler_name, scheduler,
                denoise=1.0, use_references=False, ref_image_size="match",
                ref_image=None, ref_audio=None, expressive_tts_mode="Single Voice",
                dialogue_script="", speaking_speed_hint=1.0, duration_profile="Balanced",
                ref_audio_1=None, ref_audio_2=None, speech_language="Auto",
                custom_speech_language=""):
        cfg = 1.0  # native MiniMax H3 is guidance-distilled

        # Retired labels from old workflows are routed into the unified speech mode.
        if generation_mode in ("Speech Generation", "Voice Cloning TTS"):
            generation_mode = "Expressive TTS"

        if generation_mode == "Expressive TTS" and expressive_tts_mode == "Dual Voice Dialogue":
            if ref_audio_1 is None or ref_audio_2 is None:
                raise ValueError("Expressive TTS / Dual Voice Dialogue requires both ref_audio_1 and ref_audio_2.")
            dialogue_audio, dialogue_report = MiniMaxH3DualVoiceDialogue().execute(
                model=model, clip=clip, vae=vae, audio_vae=audio_vae,
                dialogue_script=dialogue_script, speaking_speed_hint=speaking_speed_hint,
                duration_profile=duration_profile,
                speech_language=speech_language, custom_speech_language=custom_speech_language,
                steps=steps, sampler_name=sampler_name, scheduler=scheduler,
                denoise=denoise, seed=seed, ref_audio_1=ref_audio_1, ref_audio_2=ref_audio_2
            )
            return (dialogue_audio, dialogue_report)

        if generation_mode == "Speech Generation":
            script = str(speech_text or "").strip()
            effective_prompt = _build_speech_instruction(
                script,
                voice_type=voice_type,
                custom_voice_description=custom_voice_description,
                voice_timbre=voice_timbre,
                speaking_style=speaking_style,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                speech_speed=speech_speed,
                speech_pitch=speech_pitch,
            )
        elif generation_mode == "Voice Cloning TTS":
            if ref_audio is None:
                raise ValueError(
                    "Voice Cloning TTS requires an AUDIO reference connected to ref_audio."
                )
            effective_prompt = _build_voice_cloning_instruction(
                speech_text,
                voice_reference_mode=voice_reference_mode,
                speaking_style=speaking_style,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                speech_speed=speech_speed,
                additional_instructions=cloning_instructions,
            )
            # Voice Cloning TTS always uses the native reference-conditioning path.
            use_references = True
        elif generation_mode == "Expressive TTS":
            effective_prompt, expressive_segments = _build_expressive_instruction(
                speech_text,
                voice_type=voice_type,
                custom_voice_description=custom_voice_description,
                voice_timbre=voice_timbre,
                default_style=expressive_default_style,
                default_emotion=expressive_default_emotion,
                default_intensity=expressive_default_intensity,
                default_speed=expressive_default_speed,
                default_pitch=expressive_default_pitch,
                voice_reference=(ref_audio is not None),
            )
            # Optional reference audio turns Expressive TTS into reference-guided
            # expressive speech using the same native reference-conditioning path.
            effective_prompt += "\\n\\nLANGUAGE CONTROL:\\n" + _language_instruction(speech_language, custom_speech_language)
            if ref_audio is not None:
                use_references = True
        else:
            # Preserve the legacy Audio Generation prompt byte-for-byte.
            effective_prompt = prompt

        # Expressive TTS budgets duration automatically from speech text.
        # Audio Generation retains its original manual length behavior.
        if generation_mode == "Expressive TTS" and expressive_tts_mode == "Single Voice":
            spoken_text = str(speech_text or "").strip()
            if not spoken_text:
                raise ValueError("Expressive TTS requires non-empty speech_text.")
            length = _dialogue_duration_frames(spoken_text, expressive_default_speed or 1.0)

        # Target canvas for empty AV latent (audio-focused; video branch is placeholder)
        width, height = 320, 320
        latent, frame_count = _empty_av_latent(width, height, length)

        # --- conditioning ---
        if use_references and (ref_image is not None or ref_audio is not None):
            # Use generation-sized canvas metrics for 'match' scale of the ref image
            # so identity refs are sized relative to a real H3 canvas, not 320x320.
            ref_w, ref_h = 320, 320
            ref_items, ref_blocks = _build_ref2va_refs(
                vae, audio_vae, ref_w, ref_h, ref_image_size, ref_image, ref_audio
            )
            # Native tokenizer path — same as MiniMaxH3ReferenceToVideo
            tokens = clip.tokenize(effective_prompt, minimax_ref_items=ref_items)
            positive = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                positive = node_helpers.conditioning_set_values(positive, {
                    "minimax_refs": ref_blocks,
                })
        else:
            tokens = clip.tokenize(effective_prompt, images=[])
            positive = clip.encode_from_tokens_scheduled(tokens)
            # The target duration is represented by the AV latent itself.
            # Current native H3 conditioning does not require minimax_frame_count.

        negative = []

        sampler = comfy.samplers.sampler_object(sampler_name)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

        if denoise < 1.0:
            if denoise <= 0.0:
                audio_latent = latent["samples"].tensors[1]
                decoded = audio_vae.decode(audio_latent)
                return (_to_audio_dict(decoded), effective_prompt)
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

        return (audio, effective_prompt)


# -----------------------------------------------------------------------------
# Dual Voice Dialogue (0.4.0)
# -----------------------------------------------------------------------------

def _parse_dual_dialogue(script):
    """Parse [Speaker 1], [Speaker 2], optional emotion tags and [pause=...]."""
    text = str(script or "").strip()
    if not text:
        raise ValueError("Dual Voice Dialogue: dialogue_script is empty.")
    parts = re.split(r'(\[[^\]]+\])', text)
    current = None
    pending = []
    current_language = None
    result = []
    for part in parts:
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            tag = part[1:-1].strip()
            low = tag.lower()
            speaker_match = re.fullmatch(r"speaker\s*([12])(?:\s*,\s*(.*))?", tag, flags=re.IGNORECASE)
            if speaker_match:
                current = int(speaker_match.group(1))
                pending = [speaker_match.group(2).strip()] if speaker_match.group(2) else []
            elif low.startswith("lang=") or low.startswith("language="):
                current_language = tag.split("=", 1)[1].strip()
                if not current_language:
                    raise ValueError("Language override cannot be empty. Use [lang=English].")
            elif low.startswith("pause="):
                raw = tag.split("=", 1)[1].strip().lower()
                m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds)?", raw)
                if not m:
                    raise ValueError(f"Invalid [{tag}]. Use [pause=500ms] or [pause=1.2s].")
                amount = float(m.group(1)); unit = m.group(2) or "ms"
                ms = int(max(0, min(10000, amount * (1000 if unit in ("s", "sec", "seconds") else 1))))
                result.append({"kind":"pause", "duration_ms":ms})
            else:
                pending.append(tag)
        else:
            spoken = part.strip()
            if spoken:
                if current not in (1, 2):
                    raise ValueError("Every spoken line must follow [Speaker 1] or [Speaker 2].")
                result.append({"kind":"speech", "speaker":current, "directions":pending[:], "text":spoken, "language":current_language})
                pending = []
    if not any(x["kind"] == "speech" for x in result):
        raise ValueError("Dialogue contains no spoken lines.")
    return result


def _language_instruction(language="Auto", custom_language=""):
    language = str(language or "Auto").strip()
    if language == "Custom":
        language = str(custom_language or "").strip()
    if not language or language.lower() == "auto":
        return "Speech language: preserve the language of the supplied script. Do not translate or switch languages."
    return (
        f"Speech language: {language}. Speak the supplied text in this language with natural pronunciation. "
        "Preserve the original wording and script. Do not translate, invent words, or switch languages."
    )


def _count_text_syllables(text):
    """Approximate syllable count from vowel groups; language-agnostic fallback."""
    vowels = set("аеёиоуыэюяaeiouy")
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", str(text or "").lower())
    total = 0
    for word in words:
        count = sum(1 for i, char in enumerate(word)
                    if char in vowels and (i == 0 or word[i - 1] not in vowels))
        total += max(1, count)
    return max(1, total)


def _smart_duration_seconds(text, speed=1.0, profile="Balanced"):
    """Estimate natural speech duration before H3's mandatory temporal alignment."""
    text = str(text or "").strip()
    speed = max(0.5, min(2.0, float(speed)))
    syllables = _count_text_syllables(text)
    settings_by_profile = {
        "Compact": {"syllables_per_second": 5.0, "margin": 0.10, "minimum": 1.35},
        "Balanced": {"syllables_per_second": 4.2, "margin": 0.22, "minimum": 1.55},
        "Expressive": {"syllables_per_second": 3.6, "margin": 0.38, "minimum": 1.75},
    }
    settings = settings_by_profile.get(str(profile or "Balanced"), settings_by_profile["Balanced"])
    speech_seconds = syllables / (settings["syllables_per_second"] * speed)
    punctuation = min(2, text.count(",")) * 0.10
    punctuation += min(2, text.count(";") + text.count(":")) * 0.14
    if re.search(r"[.!?…]+\\s*$", text):
        punctuation += 0.16
    elif re.search(r"[.!?…]", text):
        punctuation += 0.08
    estimated = speech_seconds + punctuation + settings["margin"]
    return max(settings["minimum"], min(20.0, estimated))


def _dialogue_duration_details(text, speed=1.0, profile="Balanced"):
    estimated_seconds = _smart_duration_seconds(text, speed, profile)
    target_frames = max(5, int(math.ceil(estimated_seconds * FPS)))
    aligned_frames = align_frame_count(target_frames)
    return {
        "syllables": _count_text_syllables(text),
        "estimated_seconds": round(estimated_seconds, 3),
        "target_frames": target_frames,
        "aligned_frames": aligned_frames,
        "aligned_seconds": round(aligned_frames / FPS, 3),
        "alignment_overhead_seconds": round((aligned_frames - target_frames) / FPS, 3),
        "duration_profile": str(profile or "Balanced"),
    }


def _dialogue_duration_frames(text, speed=1.0, profile="Balanced"):
    return _dialogue_duration_details(text, speed, profile)["aligned_frames"]


def _concat_dialogue_audio(parts, sample_rate):
    if not parts:
        raise ValueError("No generated dialogue audio segments to assemble.")
    channels = max(int(p.shape[1]) for p in parts)
    normalized = []
    for wav in parts:
        wav = wav.float().cpu()
        if wav.shape[1] == 1 and channels > 1:
            wav = wav.repeat(1, channels, 1)
        elif wav.shape[1] > channels:
            wav = wav[:, :channels, :]
        normalized.append(wav)
    return {"waveform": torch.cat(normalized, dim=-1).contiguous(), "sample_rate": int(sample_rate)}


class MiniMaxH3DualVoiceDialogue:
    """Sequential two-reference dialogue generation and lossless audio assembly."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",),
                "dialogue_script": ("STRING", {"multiline": True, "default": "[Speaker 1]\nHello! How are you?\n\n[Speaker 2]\nI'm doing well, thank you!", "dynamic_prompts": False}),
                "speaking_speed_hint": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Prompt guidance and duration estimate only; not precise timing control."}),
                "duration_profile": (["Compact", "Balanced", "Expressive"], {"default": "Balanced", "tooltip": "Adaptive latent duration budget for each spoken segment."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "res_multistep"], {"default":"res_multistep"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"], {"default":"simple"}),
                "denoise": ("FLOAT", {"default":1.0, "min":0.0, "max":1.0, "step":0.01}),
                "seed": ("INT", {"default":0, "min":0, "max":0xffffffffffffffff, "step":1}),
            },
            "optional": {
                "ref_audio_1": ("AUDIO", {"tooltip":"Voice identity reference for Speaker 1."}),
                "ref_audio_2": ("AUDIO", {"tooltip":"Voice identity reference for Speaker 2."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("dialogue_audio", "segment_report")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/minimax"

    def execute(self, model, clip, vae, audio_vae, dialogue_script, speaking_speed_hint,
                steps, sampler_name, scheduler, denoise, seed,
                ref_audio_1=None, ref_audio_2=None, speech_language="Auto",
                custom_speech_language="", duration_profile="Balanced"):
        if ref_audio_1 is None or ref_audio_2 is None:
            raise ValueError("Dual Voice Dialogue requires both ref_audio_1 and ref_audio_2.")
        segments = _parse_dual_dialogue(dialogue_script)
        cfg = 1.0
        sampler = comfy.samplers.sampler_object(sampler_name)
        model_sampling = model.get_model_object("model_sampling")
        base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        seed_base = int(seed) & 0xffffffffffffffff
        generated = []
        report = []
        cursor = 0
        for index, seg in enumerate(segments):
            if seg["kind"] == "pause":
                ms = seg["duration_ms"]
                if generated and ms:
                    sr = generated[-1].shape[-1]  # overwritten below; only used to avoid assumptions
                    sample_rate = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
                    silence = torch.zeros((1, 1, round(sample_rate * ms / 1000.0)), dtype=torch.float32)
                    generated.append(silence)
                report.append({"index":index+1,"type":"pause","duration_ms":ms})
                continue
            speaker = seg["speaker"]
            ref = ref_audio_1 if speaker == 1 else ref_audio_2
            directions = ", ".join(seg["directions"]) if seg["directions"] else "natural conversational delivery"
            duration_details = _dialogue_duration_details(
                seg["text"], speaking_speed_hint, duration_profile
            )
            duration_frames = duration_details["aligned_frames"]
            line_language = seg.get("language") or speech_language
            language_rule = _language_instruction(
                line_language,
                custom_speech_language if str(line_language).strip() == "Custom" else ""
            )
            prompt = (
                "Generate spoken dialogue audio for exactly one speaker.\n"
                f"{language_rule}\n"
                "<Audio 1> is the PRIMARY and exclusive voice identity reference for this utterance. "
                "Preserve its recognizable vocal identity, perceived age, vocal gender presentation, accent, timbre, resonance and articulation as closely as the model allows. "
                "Do not imitate or introduce another speaker. Do not reproduce the reference recording's words or background sounds.\n"
                f"Delivery direction: {directions}.\n"
                f"Speaking pace: approximately {speaking_speed_hint:.2f}x, as a soft prompt hint.\n"
                "Speak only the exact text below, in its original language. Do not paraphrase, translate, omit, or add words. "
                "No music or sound effects.\n\n"
                f"DIALOGUE: {seg['text']}"
            )
            width = height = 320
            latent, _ = _empty_av_latent(width, height, duration_frames)
            ref_items, ref_blocks = _build_ref2va_refs(vae, audio_vae, width, height, "match", None, ref)
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            positive = clip.encode_from_tokens_scheduled(tokens)
            positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": ref_blocks})
            sigmas = base_sigmas
            if denoise < 1.0:
                if denoise <= 0.0:
                    raise ValueError("denoise must be greater than 0 for Dual Voice Dialogue.")
                total_steps = max(1, int(steps * denoise))
                sigmas = sigmas[len(sigmas) - total_steps - 1:]
            chunk_seed = (seed_base + index * 1000003) & 0xffffffffffffffff
            noise = comfy.sample.prepare_noise(latent["samples"], chunk_seed)
            try:
                samples = comfy.sample.sample_custom(
                    model=model, noise=noise, cfg=cfg, sampler=sampler, sigmas=sigmas,
                    positive=positive, negative=[], latent_image=latent["samples"],
                    noise_mask=None, callback=None, disable_pbar=False, seed=chunk_seed)
                audio_latent = samples.tensors[1] if hasattr(samples, "tensors") else samples["samples"].tensors[1] if isinstance(samples, dict) else samples
                decoded = _to_audio_dict(audio_vae.decode(audio_latent))
            except Exception as exc:
                raise RuntimeError(f"Dual Voice Dialogue failed at segment {index+1} (Speaker {speaker}): {exc}") from exc
            wav = decoded["waveform"]
            generated.append(wav)
            report.append({
                "index": index + 1, "type": "speech", "speaker": speaker,
                "text": seg["text"], "directions": seg["directions"],
                "language": line_language, **duration_details,
                "estimated_frames": duration_details["aligned_frames"],
                "sample_rate": decoded["sample_rate"], "samples": int(wav.shape[-1])
            })
        # Resample all chunks to a common sample rate before concatenation.
        target_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
        fixed = []
        for chunk in generated:
            # Silence chunks were created at target_sr; generated chunks carry their own SR below.
            fixed.append(chunk)
        # Speech decode is normally at the model's audio sample rate; all native chunks share it.
        audio = _concat_dialogue_audio(fixed, target_sr)
        return (audio, json.dumps({
            "segments": report,
            "note": "Adaptive duration is a latent time budget, not a guaranteed speech stop boundary. MiniMax H3 may still add or omit words; validate with listening or ASR."
        }, ensure_ascii=False, indent=2))
