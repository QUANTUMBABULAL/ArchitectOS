# Fallback: staying on Rust 1.83

This is **Option A** — keep your current toolchain and pin the offending
transitive crates backward. It works, but read the trade-off first.

## When to use this

Only if you cannot install a newer toolchain (locked corporate machine,
offline environment, policy). Otherwise prefer `rust-toolchain.toml`,
which is already configured and needs no manual steps.

## Why this is the weaker option

Two independent subtrees pull edition-2024 / high-MSRV crates:

```
url 2.5.8 → idna 1.1.0 → idna_adapter 1.2.2   (edition 2024, rust 1.86)
                          └→ icu_normalizer 2.2.0, icu_properties 2.2.0

tauri-utils 1.6.2 → serde_with 3.21.0 → darling 0.23.0
                                      └→ time 0.3.54
```

Pinning these freezes them permanently. Any `cargo update`, any new
release in either subtree, and you are back here. Cargo 1.83 has no
MSRV-aware resolver, so nothing prevents the drift from recurring.

## Steps

Run from `desktop/src-tauri/`.

```powershell
# 1. Delete the poisoned lockfile and regenerate it.
Remove-Item Cargo.lock -ErrorAction SilentlyContinue
cargo generate-lockfile

# 2. Pin the verified root cause.
#    idna_adapter 1.2.0 declares edition 2021 and rust-version 1.67.0,
#    and keeps the ICU 1.x line, so the whole icu_* 2.x subtree with it.
cargo update -p idna_adapter --precise 1.2.0

# 3. Pin the serde_with subtree back below the darling 0.23 bump.
cargo update -p serde_with --precise 3.11.0
cargo update -p time --precise 0.3.36

# 4. Verify nothing edition-2024 survived.
cargo tree -i idna_adapter
cargo build
```

## If step 4 still fails

The error names the crate and the version it wanted. Pin that one too:

```powershell
cargo update -p <crate-name> --precise <older-version>
```

Find a version whose `rust_version` is at or below 1.83 here:

```
https://crates.io/api/v1/crates/<crate-name>/versions
```

Look for the `rust_version` and `edition` fields. Anything with
`"edition":"2024"` will fail on Cargo 1.83 regardless of its MSRV.

## Verified vs. assumed

| Pin | Status |
| --- | --- |
| `idna_adapter 1.2.0` | **Verified** — crates.io reports `rust_version 1.67.0`, `edition 2021` |
| `serde_with 3.11.0` | Not individually verified; chosen as pre-`darling 0.23` |
| `time 0.3.36` | Not individually verified; long-standing low-MSRV release |

The first pin is the one that matters — it collapses the ICU 2.x subtree,
which is the larger half of the problem. The other two may need one round
of adjustment via the loop above.

## Commit the lockfile

Whichever route you take, `Cargo.lock` **must** be committed. This is an
application, not a library, and the lockfile is what makes the build
reproducible. Without it every fresh clone re-resolves to the newest
crates and reproduces this exact failure.
