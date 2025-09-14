# bias-rs

`bias-rs` is a Rust library and CLI for checking dataset bias in tabular data.

The project focuses on dataset composition, not model behavior. The first
implementation targets:

- CSV ingestion
- Parquet ingestion
- Configurable bias detectors
- JSON and plain-text reports

The codebase is organized as a small workspace:

- `crates/bias-rs`: library crate
- `crates/bias-cli`: command-line interface

## Development

The expected local workflow is:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```
