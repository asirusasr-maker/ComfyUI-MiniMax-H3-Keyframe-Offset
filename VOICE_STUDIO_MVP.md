# Voice Studio MVP — implementation notes

## What's new
- Added `Audio Generation`, `Speech Generation`, and `Voice Cloning TTS` modes to the existing AIO node.
- Audio Generation keeps the previous prompt-conditioning path unchanged.
- Speech Generation builds a plain-text instruction from the supplied script, voice description,
  timbre, speaking style, emotion, intensity, pace, and pitch direction.
- Added `conditioning_prompt` as a second output; the original `audio` output remains first.
- Frontend visibility switches between audio and speech controls.
- The existing node class ID remains `MiniMaxH3AudioGenerator` for workflow compatibility.
- No global monkey patch was introduced.

## Important limitations
- Voice type, timbre, emotion, speed, and pitch are prompt-level hints only.
- This MVP does not implement verified zero-shot TTS, voice cloning, speaker identity locking,
  or deterministic acoustic controls. Results depend on the installed MiniMax H3 checkpoint
  and its text/audio conditioning behavior.
- Reference-audio inputs remain part of the existing Audio Generation/Ref2VA path. The existing reference path remains available in either mode; dedicated Voice Cloning TTS
  with validated speaker-identity behavior will be designed and tested separately.

## Install
Replace the existing custom-node folder with the contents of this archive, restart ComfyUI,
and test both modes. Keep a backup of your current folder/workflows.


## Voice Cloning TTS (experimental)
- Requires `speech_text` and an `AUDIO` connection to the `ref_audio` socket.
- Automatically enables native MiniMax H3 reference conditioning and inserts `<Audio 1>`
  in the generated instruction.
- The reference is encoded with the existing Audio VAE path and attached through
  `minimax_ref_items` / `minimax_refs`, matching the upstream standalone-audio reference flow.
- `voice_reference_mode`, speaking style, emotion, intensity, speed, and additional instructions
  are prompt-level guidance, not guaranteed acoustic controls.
- This is an experimental reference-conditioned speech workflow. The upstream H3 reference
  path supports standalone audio references, but that alone does not prove robust arbitrary-text
  zero-shot voice cloning for every checkpoint. Test with short clean speech references first.


## Expressive TTS (0.3.0)
- Adds an `Expressive TTS` mode with inline stage directions in `speech_text`.
- Supported examples: `[happy]`, `[whispering, sad]`, `[laughing]`, `[crying]`,
  `[sighing]`, `[shouting]`, and timed pauses such as `[pause=500ms]` or `[pause=1.2s]`.
- Default style, emotion, intensity, speed, and pitch direction are included in the
  generated conditioning prompt.
- An optional `ref_audio` can guide the voice identity using the existing native reference path.
- Inline directions are prompt-level guidance; the model may not render every event or pause
  exactly. This mode currently creates one prompt/one generation, not separately synthesized
  chunks or guaranteed timing-accurate segments.


## Expressive TTS 0.3.1 — Voice Identity and Performance
- Expressive TTS now uses the shared `voice_type`, `custom_voice_description`, and
  `voice_timbre` widgets as its Voice Identity controls.
- `expressive_default_style`, `expressive_default_emotion`,
  `expressive_default_intensity`, `expressive_default_speed`, and
  `expressive_default_pitch` remain the only default performance controls in this mode.
- Duplicate generic controls (`speaking_style`, `emotion`, `emotion_intensity`,
  `speech_speed`, `speech_pitch`) are hidden in Expressive TTS to avoid ambiguity.
- Without `ref_audio`, the prompt includes the selected voice identity and timbre.
- With `ref_audio`, `<Audio 1>` is explicitly designated as the primary identity source;
  conflicting generic voice-type instructions are not applied.
- Existing widget names are retained to preserve saved workflow values.
