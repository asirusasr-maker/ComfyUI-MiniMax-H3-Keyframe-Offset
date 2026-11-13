# Voice Studio 0.5.3 — Language Control

- Added shared Speech Language selector to Expressive TTS: Auto, English, Russian, Chinese, Japanese, Korean, Spanish, French, German, Portuguese, Italian, Custom.
- Added Custom Language text field, shown only when Custom is selected.
- Single Voice receives an explicit language instruction in its conditioning prompt.
- Dual Voice Dialogue uses the shared default language, with per-line overrides using [lang=English] or [language=Russian].
- Language override remains active for subsequent utterances until another language tag is supplied.
- Segment report includes the effective language.
- No changes to Audio Generation, Smart Duration Engine, voice reference identity logic, or audio assembly.
- Based on validated 0.5.1 baseline plus 0.5.2 Tag Picker.
- Language guidance is prompt-level and cannot guarantee perfect pronunciation or prevent all model language drift.
