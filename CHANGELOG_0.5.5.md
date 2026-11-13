# 0.5.5 — Adaptive Speech Duration Engine 2.0

- Added Compact / Balanced / Expressive duration profiles for Dual Voice Dialogue.
- Replaced word-count-only estimation with approximate syllable-based estimation, punctuation allowances, speaking-speed scaling, and profile-specific margins.
- Kept required MiniMax H3 temporal alignment; the segment report exposes estimated seconds, pre-alignment target frames, aligned frames, alignment overhead, syllable count, and selected profile.
- Kept sequential speaker-reference generation, language overrides, pause assembly, Tag Picker, and socket visibility behavior from 0.5.4.1.
- Duration is a latent budget, not a guaranteed stop boundary. It can reduce excess time but cannot guarantee verbatim speech or prevent all model continuation.
