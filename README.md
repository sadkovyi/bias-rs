# bias-rs

`bias-rs` is a Rust library and CLI for checking bias in tabular datasets.

The project is about dataset composition. It does not inspect model outputs,
predictions, or fairness metrics tied to a classifier. Given a dataset plus one
or more sensitive columns, it looks for patterns such as group imbalance,
missingness differences, categorical distribution shifts, and numeric
distribution shifts.

## Current scope

The library can:

- read CSV files
- read Parquet files
- audit one sensitive column at a time
- audit intersectional groupings
- emit JSON reports
- emit plain-text CLI reports
- apply multiple-testing correction across statistical findings

The library does not:

- make causal claims
- infer what the "right" population mix should be unless you provide a
  reference distribution
- analyze model behavior

## Detectors

`bias-rs` currently ships four detector families.

1. Representation checks report group counts, proportions, entropy, and large
   imbalances. If you supply an expected distribution, the library also runs a
   goodness-of-fit test.
2. Missingness checks compare null rates across sensitive groups.
3. Categorical association checks compare sensitive groups against categorical
   feature values with a contingency-table test.
4. Numeric distribution checks compare numeric feature distributions with
   rank-based tests.

By default, statistical findings are filtered with Benjamini-Hochberg
correction.

## CLI quick start

Build the CLI:

```bash
cargo build -p bias-cli
```

Run an audit against a CSV file:

```bash
cargo run -p bias-cli -- audit \
  --input data/employees.csv \
  --format csv \
  --sensitive gender \
  --sensitive race \
  --grouping both \
  --output text
```

Ask for JSON instead:

```bash
cargo run -p bias-cli -- audit \
  --input data/employees.csv \
  --format csv \
  --sensitive gender \
  --output json
```

Read Parquet instead of CSV:

```bash
cargo run -p bias-cli -- audit \
  --input data/employees.parquet \
  --format parquet \
  --sensitive gender \
  --output text
```

Limit the audit to a subset of columns:

```bash
cargo run -p bias-cli -- audit \
  --input data/employees.csv \
  --format csv \
  --sensitive gender \
  --columns age,region,tenure \
  --detector representation \
  --detector numeric-distribution
```

## Expected distributions

If you want representation checks to compare the dataset against a known
baseline, pass a JSON file with group proportions keyed by grouping name.

Example file:

```json
{
  "gender": {
    "woman": 0.51,
    "man": 0.49
  },
  "gender+race": {
    "gender=woman|race=asian": 0.12,
    "gender=woman|race=white": 0.39,
    "gender=man|race=asian": 0.11,
    "gender=man|race=white": 0.38
  }
}
```

Use it from the CLI:

```bash
cargo run -p bias-cli -- audit \
  --input data/employees.csv \
  --format csv \
  --sensitive gender \
  --expected-dist expected.json
```

## Library example

```rust
use bias_rs::{AuditConfig, CsvReadOptions, audit_dataset, read_csv};

let dataset = read_csv("data/employees.csv", CsvReadOptions::default())?;
let config = AuditConfig::builder()
    .sensitive_column("gender")
    .min_group_size(20)
    .build();
let report = audit_dataset(&dataset, &config)?;

println!("rows: {}", report.dataset.row_count);
println!("findings: {}", report.findings.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Report shape

An `AuditReport` includes:

- dataset metadata
- effective audit configuration
- detector run summaries
- per-group summaries
- skipped analyses
- final findings, including raw and corrected p-values when a statistical test
  was used

The CLI prints the same report either as JSON or as a readable text summary.

More detail is in:

- `docs/methodology.md`
- `docs/report-schema.md`

## Development

Useful commands while working on the project:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Notes on interpretation

This library can tell you that groups are imbalanced or that feature
distributions differ across groups. It cannot tell you why those differences
exist, whether they are legally problematic, or whether a downstream model will
amplify them. Those questions need domain context.
