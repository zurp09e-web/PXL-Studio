# PXL Studio 6.0 — adaptive spatial entropy

- New writer version: PXL v6; v1-v5 reading retained.
- Added codec 24: 17-model signed-small-residual contexts.
- Added codec 25: 65-model left/up/upper-left contexts, used in maximum effort.
- Encoder fully serializes candidate payloads and adopts a new context model
  only when it is smaller than the original candidate.
- Added residual-distribution screening to avoid needless signed-context work
  on texture/noise-like tiles.
- Native C++ ABI bumped from 3 to 4 with signed and triple-context rANS decoders.
- Python fallback supports both new context models.
- Existing PXL v4/v5 fixtures remain decodable.
