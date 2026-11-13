# Voice Studio 0.5.4 — Dual Voice Socket Layout

- Dual Voice Dialogue now displays only the two dedicated speaker audio sockets; the single-voice `ref_audio` and image `ref_image` sockets are hidden in this submode.
- The two dialogue reference sockets are presented top-to-bottom as `speaker_1` and `speaker_2`.
- The canonical backend socket names (`ref_audio_1`, `ref_audio_2`) are retained internally to avoid changing the Python execution contract and existing workflow/API input keys.
- Outside Dual Voice Dialogue, the single-voice audio and image reference sockets retain their existing mode-dependent visibility.
- No changes to dialogue parsing, language control, Tag Picker, Smart Duration Engine, voice reference selection, sampling, or audio assembly.
- Built from 0.5.3 Language Control.
