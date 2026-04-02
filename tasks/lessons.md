# Lessons

- 2026-03-27: When fixing delivery-blocking regressions, prefer root-cause changes that remove the bad dependency or contract mismatch. Do not ship temporary fallbacks, compatibility shims, or error-swallowing patches as a substitute for fixing the underlying bug.
- 2026-03-31: When a user reports an error from another machine, explicitly separate locally verified facts from inferences about the user's environment. Do not phrase local reproduction details as if they were observed on the user's device.
- 2026-03-31: When narrowing a policy change based on an example, confirm whether the user wants that behavior only for the example path or for the whole action family before implementation.
- 2026-04-01: When a user corrects the failing layer of a model/provider bug, re-anchor on that exact runtime stage before patching. Distinguish config-generation defaults from runtime handshake/discovery metadata, and verify which layer is actually producing the bad fallback.
- 2026-04-02: When a user reports a mobile screen still feels cramped or visually weak after a first pass, revisit the root container and information hierarchy together. Do not treat it as only a per-screen spacing tweak when the app-level frame or layout style may still be wrong.
