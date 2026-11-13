# 0.5.4.1 — Socket Visibility Correction

- Fixed socket visibility: inactive optional inputs are removed from the visible node layout and restored when the corresponding mode is selected.
- Dual Voice Dialogue displays only `speaker_1` and `speaker_2`, backed by the original `ref_audio_1` and `ref_audio_2` input names.
- Single Voice displays only `ref_audio`.
- Audio Generation displays reference sockets only when `use_references` is enabled.
- Existing upstream connections are retained while sockets are temporarily removed and restored during mode changes.
- No changes to generation, language control, tag picker, Smart Duration Engine, or voice identity logic.

Runtime verification in ComfyUI is still required.
