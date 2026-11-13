# Voice Studio 0.5.0

- Only Audio Generation and Expressive TTS remain as top-level modes.
- Expressive TTS contains Single Voice and Dual Voice Dialogue submodes.
- Dual dialogue runs inside the AIO node using two independent audio references and sequential utterance generation.
- Smart Duration Engine estimates duration using word count, average word length, punctuation, speed and a safety margin.
- Single Voice Expressive TTS uses automatic duration estimation; Audio Generation keeps manual length.
- Standalone Dual Voice Dialogue node is no longer registered.
- No global monkey patches.
