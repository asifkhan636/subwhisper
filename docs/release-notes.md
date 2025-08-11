# Release Notes

## Unreleased
- Upgrade dependencies to drop `pkg_resources` usage:
  - `pyannote.audio>=3.3.2` now depends on `lightning` instead of `pytorch-lightning`.
  - Require `torch>=2.5.1` and `lightning>=2.5.2` to align with the new stack.
