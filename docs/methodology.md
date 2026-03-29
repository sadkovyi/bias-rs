# Methodology

`bias-rs` looks for signs of dataset bias in tabular data. It does not inspect
model behavior and it does not require labels or predictions.

## What the library checks

The current library ships four detector families:

1. Representation checks count how many rows belong to each sensitive group and
   report large imbalances. If you provide a reference distribution, the library
   also runs a goodness-of-fit test against that baseline.
2. Missingness checks compare null rates for each analyzed column across
   sensitive groups.
3. Categorical association checks compare group membership against categorical
   feature values with a contingency-table test.
4. Numeric distribution checks compare numeric feature distributions with
   rank-based tests.

## What counts as a finding

Representation findings without a reference distribution are descriptive. They
flag groups that are much smaller than the largest observed group, but they do
not claim that the dataset is biased relative to any outside population.

Statistical findings for missingness, categorical association, and numeric
distribution are filtered through the configured significance threshold and
multiple-testing correction. By default, the library uses
Benjamini-Hochberg correction.

Severity is a separate step. Once a finding survives the p-value filter,
`bias-rs` assigns `warning` or `critical` with detector-specific thresholds.

## Severity thresholds

- Representation uses ratios, not p-values. By default, a group is a warning
  below `0.8` of the largest or expected group and critical below `0.5`.
  These are project defaults meant to surface large representation gaps.
- Missingness uses the largest minus smallest missing-rate gap. By default, a
  significant result becomes critical at `0.25`. This is a project heuristic.
- Categorical association uses Cramer's V. By default, a significant result
  becomes critical at `0.3`. This is a common rule of thumb for at least a
  moderate association.
- Numeric distribution with two groups uses Cliff's delta. By default, a
  significant result becomes critical at `0.33`. This matches a common
  medium-effect cutoff.
- Numeric distribution with three or more groups uses epsilon-squared. By
  default, a significant result becomes critical at `0.26`. This is a project
  default meant to highlight strong rank separation across groups.

All of these thresholds are configurable through detector settings in the
library API and through matching CLI flags.

## Caveats

- Small groups are skipped for statistical testing once they fall under the
  configured minimum group size.
- A significant difference is not the same as a causal explanation.
- Group labels come from the dataset as-is. If the source data is inconsistent,
  the report will reflect that inconsistency.
- Representation claims against a real-world population need a reference
  distribution supplied by the caller.
