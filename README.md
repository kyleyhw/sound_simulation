# release-assets

Binary inputs for the `Data release` workflow (`.github/workflows/release-data.yml`
on `main`). Only files that cannot be regenerated from `data/MANIFEST.json`
live here: the skip_v2 checkpoints (plan 3.4.7). Their SHA-256 values are in
the manifest, and the workflow refuses to publish on a mismatch.
