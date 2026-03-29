# bias-rs

`bias-rs` is a Rust library and CLI for auditing tabular datasets, especially
the metadata tables that sit around AI/ML training and evaluation data.

The project is about dataset composition. It does not inspect model outputs,
predictions, or fairness metrics tied to a classifier. Given a CSV or Parquet
table plus one or more sensitive columns, it looks for group imbalance,
missingness differences, categorical distribution shifts, and numeric
distribution shifts.

## Current scope

The library can:

- read CSV and Parquet metadata tables
- audit one sensitive column at a time or intersectionally
- check representation, missingness, and feature shifts across groups
- emit JSON reports for automation
- emit plain-text CLI reports
- apply multiple-testing correction across statistical findings

The library does not:

- make causal claims
- infer what the "right" population mix should be unless you provide a
  reference distribution
- inspect model outputs, predictions, or fairness metrics

## Detectors

`bias-rs` currently ships four detector families.

1. Representation checks report group counts, proportions, entropy, and large
   imbalances. If you supply an expected distribution, the library also runs a
   goodness-of-fit test.
2. Missingness checks compare null rates across sensitive groups.
3. Categorical association checks compare sensitive groups against categorical
   feature values such as `label`, `source`, or `split` with a
   contingency-table test.
4. Numeric distribution checks compare numeric feature distributions such as
   `token_count`, `duration_seconds`, or `age` with rank-based tests.

By default, statistical findings are filtered with Benjamini-Hochberg
correction.

## Severity thresholds

Each detector has its own severity cutoff. A finding must first survive the
configured p-value filter. After that, `bias-rs` marks it as `warning` or
`critical` with detector-specific thresholds.

Default thresholds:

- representation: `warning_ratio = 0.8`, `critical_ratio = 0.5`
- missingness: `critical_rate_gap = 0.25`
- categorical association: `critical_cramers_v = 0.3`
- numeric distribution, two groups: `critical_cliffs_delta = 0.33`
- numeric distribution, three or more groups: `critical_epsilon_squared = 0.26`

Override them from the CLI when you need a different review bar:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.csv \
  --format csv \
  --sensitive gender \
  --columns label,source,token_count \
  --missingness-critical-rate-gap 0.20 \
  --categorical-critical-cramers-v 0.40 \
  --numeric-critical-cliffs-delta 0.45
```

## Where this fits in ML workflows

- check the composition of a training or evaluation dataset before model work
- spot missingness, source skew, or label skew across sensitive groups
- emit JSON reports for CI jobs or recurring dataset review steps

For text, image, audio, or multimodal corpora, the expected input is still a
CSV or Parquet metadata table rather than the raw assets themselves.

## CLI quick start

Build the CLI:

```bash
cargo build -p bias-cli
```

Run an audit against a CSV file:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.csv \
  --format csv \
  --sensitive gender \
  --sensitive age_bucket \
  --grouping both \
  --columns label,source,token_count \
  --output text
```

Ask for JSON instead:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.csv \
  --format csv \
  --sensitive gender \
  --columns label,source,token_count \
  --output json
```

Read Parquet instead of CSV:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.parquet \
  --format parquet \
  --sensitive gender \
  --columns label,source,token_count \
  --output text
```

Limit the audit to a subset of columns:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.csv \
  --format csv \
  --sensitive gender \
  --columns label,source,token_count \
  --detector representation \
  --detector numeric-distribution
```

## Expected distributions

If you want representation checks to compare a training or evaluation set
against a target mix, pass a JSON file with group proportions keyed by grouping
name. That target might come from a benchmark spec, an annotation plan, or an
external baseline.

Example file:

```json
{
  "gender": {
    "woman": 0.5,
    "man": 0.5
  },
  "gender+age_bucket": {
    "gender=woman|age_bucket=18-34": 0.26,
    "gender=woman|age_bucket=35-54": 0.24,
    "gender=man|age_bucket=18-34": 0.25,
    "gender=man|age_bucket=35-54": 0.25
  }
}
```

Use it from the CLI:

```bash
cargo run -p bias-cli -- audit \
  --input data/training_metadata.csv \
  --format csv \
  --sensitive gender \
  --sensitive age_bucket \
  --grouping both \
  --expected-dist expected.json
```

## Library example

```rust
use bias_rs::{
    AuditConfig, ColumnSelection, DetectorConfig, GroupingMode, MissingnessConfig,
    ParquetReadOptions, audit_dataset, read_parquet,
};

let dataset = read_parquet(
    "data/training_metadata.parquet",
    ParquetReadOptions::default(),
)?;
let config = AuditConfig::builder()
    .sensitive_columns(["gender", "age_bucket"])
    .grouping_mode(GroupingMode::Both)
    .analysis_columns(ColumnSelection::Named(vec![
        "label".into(),
        "source".into(),
        "token_count".into(),
    ]))
    .detector(DetectorConfig::Missingness(MissingnessConfig {
        critical_rate_gap: 0.2,
        ..MissingnessConfig::default()
    }))
    .min_group_size(50)
    .build();
let report = audit_dataset(&dataset, &config)?;

println!("rows: {}", report.dataset.row_count);
println!("findings: {}", report.findings.len());
Ok::<(), Box<dyn std::error::Error>>(())
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
The JSON form is easy to feed into automated dataset checks.

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
