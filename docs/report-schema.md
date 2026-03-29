# Report schema

The library returns an `AuditReport` value. The CLI can print that report as
JSON or as plain text.

## Top-level fields

- `dataset`: row count, column count, batch count, and per-column metadata.
- `config`: the effective sensitive columns, grouping mode, alpha, and
  multiple-testing strategy, plus the detector settings used for the run.
- `detector_runs`: one entry per detector and grouping combination.
- `group_summaries`: row counts and proportions for each observed group.
- `skipped`: analyses that were not run, along with the reason.
- `findings`: the final set of findings after p-value correction.

## Config fields

The `config.detectors` array stores the effective detector settings. That
includes severity thresholds such as `critical_rate_gap`,
`critical_cramers_v`, `critical_cliffs_delta`, and
`critical_epsilon_squared`.

## Finding fields

Each finding contains:

- `detector`: which detector produced the finding
- `grouping`: which sensitive grouping was analyzed
- `sensitive_columns`: the columns that define that grouping
- `target_column`: the analyzed feature column, when the detector works on a
  specific feature
- `group`: an optional group label for group-specific findings
- `severity`: `info`, `warning`, or `critical`
- `message`: short human-readable description
- `p_value`: raw p-value when a statistical test was used
- `corrected_p_value`: adjusted p-value after multiple-testing correction
- `effect_size`: detector-specific effect size when available
- `metrics`: detector-specific numeric details
