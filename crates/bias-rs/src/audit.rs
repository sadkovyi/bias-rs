use std::collections::BTreeMap;

use crate::config::{DetectorConfig, DetectorKind, RepresentationConfig};
use crate::report::{AuditReportConfig, DetectorRun, GroupSummary};
use crate::stats::{goodness_of_fit, normalized_entropy};
use crate::table::{GroupingSpec, build_group_keys, grouping_specs};
use crate::{AuditConfig, AuditReport, BiasError, ColumnProfile, Dataset, DatasetSummary, Finding};
use crate::{Severity, SkippedAnalysis};

/// Runs a dataset audit.
pub fn audit_dataset(dataset: &Dataset, config: &AuditConfig) -> Result<AuditReport, BiasError> {
    if config.sensitive_columns.is_empty() {
        return Err(BiasError::InvalidConfig(
            "at least one sensitive column is required".to_string(),
        ));
    }

    let missing_columns: Vec<_> = config
        .sensitive_columns
        .iter()
        .filter(|column| !dataset.has_column(column))
        .cloned()
        .collect();
    if let Some(column) = missing_columns.first() {
        return Err(BiasError::MissingColumn(column.clone()));
    }

    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(BiasError::InvalidConfig(
            "alpha must be between 0.0 and 1.0".to_string(),
        ));
    }

    if config
        .reference_distributions
        .values()
        .any(|distribution| distribution.groups.values().any(|value| *value <= 0.0))
    {
        return Err(BiasError::InvalidConfig(
            "reference distributions must contain positive proportions".to_string(),
        ));
    }

    let grouping_specs = grouping_specs(config);
    let representation_config = config
        .detectors
        .iter()
        .find_map(|detector| match detector {
            DetectorConfig::Representation(settings) => Some(settings.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let mut detector_runs = Vec::new();
    let mut group_summaries = Vec::new();
    let mut findings = Vec::new();
    let mut skipped = Vec::new();
    for grouping in &grouping_specs {
        let group_keys = build_group_keys(dataset, grouping)?;
        let counts = count_groups(&group_keys);
        if counts.is_empty() {
            skipped.push(SkippedAnalysis {
                detector: DetectorKind::Representation,
                grouping: grouping.label.clone(),
                target_column: None,
                reason: "no rows were available for analysis".to_string(),
            });
            continue;
        }

        let summaries = build_group_summaries(
            grouping,
            &counts,
            config.reference_distributions.get(&grouping.label),
        );
        let baseline = config.reference_distributions.get(&grouping.label);
        let mut grouping_findings =
            representation_findings(grouping, &counts, baseline, &representation_config, config.alpha);
        findings.append(&mut grouping_findings);
        group_summaries.extend(summaries);
        detector_runs.push(DetectorRun {
            detector: DetectorKind::Representation,
            grouping: grouping.label.clone(),
            analyzed_columns: 1,
            finding_count: findings
                .iter()
                .filter(|finding| {
                    finding.detector == DetectorKind::Representation
                        && finding.grouping == grouping.label
                })
                .count(),
        });
    }

    findings.sort_by(|left, right| {
        left.grouping
            .cmp(&right.grouping)
            .then_with(|| left.group.cmp(&right.group))
            .then_with(|| left.target_column.cmp(&right.target_column))
    });

    Ok(AuditReport {
        dataset: DatasetSummary {
            row_count: dataset.row_count(),
            column_count: dataset.column_count(),
            batch_count: dataset.batches().len(),
            columns: dataset
                .schema()
                .fields()
                .iter()
                .map(|field| ColumnProfile {
                    name: field.name().to_string(),
                    data_type: field.data_type().to_string(),
                    nullable: field.is_nullable(),
                })
                .collect(),
        },
        config: AuditReportConfig {
            sensitive_columns: config.sensitive_columns.clone(),
            grouping_mode: config.grouping_mode,
            alpha: config.alpha,
            multiple_testing: config.multiple_testing,
        },
        detector_runs,
        group_summaries,
        skipped,
        findings,
    })
}

fn count_groups(group_keys: &[String]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for group_key in group_keys {
        *counts.entry(group_key.clone()).or_insert(0) += 1;
    }
    counts
}

fn build_group_summaries(
    grouping: &GroupingSpec,
    counts: &BTreeMap<String, usize>,
    baseline: Option<&crate::ReferenceDistribution>,
) -> Vec<GroupSummary> {
    let total = counts.values().sum::<usize>() as f64;
    counts
        .iter()
        .map(|(group, count)| GroupSummary {
            grouping: grouping.label.clone(),
            sensitive_columns: grouping.columns.clone(),
            group: group.clone(),
            row_count: *count,
            proportion: if total > 0.0 {
                *count as f64 / total
            } else {
                0.0
            },
            expected_proportion: baseline.and_then(|distribution| distribution.groups.get(group).copied()),
        })
        .collect()
}

fn representation_findings(
    grouping: &GroupingSpec,
    counts: &BTreeMap<String, usize>,
    baseline: Option<&crate::ReferenceDistribution>,
    config: &RepresentationConfig,
    alpha: f64,
) -> Vec<Finding> {
    let total = counts.values().sum::<usize>();
    if total == 0 {
        return Vec::new();
    }

    let max_count = counts.values().copied().max().unwrap_or(0) as f64;
    let min_count = counts.values().copied().min().unwrap_or(0) as f64;
    let total_f = total as f64;
    let entropy = normalized_entropy(&counts.values().copied().collect::<Vec<_>>());
    let imbalance_ratio = if max_count > 0.0 {
        min_count / max_count
    } else {
        1.0
    };

    let mut findings = Vec::new();
    if let Some(distribution) = baseline {
        let observed = counts.values().copied().map(|count| count as u64).collect::<Vec<_>>();
        let expected = counts
            .keys()
            .filter_map(|group| distribution.groups.get(group).copied())
            .collect::<Vec<_>>();
        if expected.len() == observed.len() {
            if let Some(test) = goodness_of_fit(&observed, &expected) {
                if test.p_value <= alpha {
                    findings.push(Finding {
                        detector: DetectorKind::Representation,
                        grouping: grouping.label.clone(),
                        sensitive_columns: grouping.columns.clone(),
                        target_column: None,
                        group: None,
                        severity: if imbalance_ratio < config.critical_ratio {
                            Severity::Critical
                        } else {
                            Severity::Warning
                        },
                        message: "group proportions differ from the configured reference distribution"
                            .to_string(),
                        p_value: Some(test.p_value),
                        corrected_p_value: None,
                        effect_size: None,
                        metrics: BTreeMap::from([
                            ("chi_square".to_string(), test.statistic),
                            (
                                "min_expected_count".to_string(),
                                test.min_expected_count,
                            ),
                            ("imbalance_ratio".to_string(), imbalance_ratio),
                            ("group_count".to_string(), counts.len() as f64),
                        ]),
                    });
                }
            }
        }

        for (group, count) in counts {
            if let Some(expected_proportion) = distribution.groups.get(group) {
                let observed_proportion = *count as f64 / total_f;
                let ratio = observed_proportion / expected_proportion;
                if ratio < config.warning_ratio {
                    findings.push(Finding {
                        detector: DetectorKind::Representation,
                        grouping: grouping.label.clone(),
                        sensitive_columns: grouping.columns.clone(),
                        target_column: None,
                        group: Some(group.clone()),
                        severity: if ratio < config.critical_ratio {
                            Severity::Critical
                        } else {
                            Severity::Warning
                        },
                        message: format!(
                            "group `{group}` appears underrepresented relative to the reference distribution"
                        ),
                        p_value: None,
                        corrected_p_value: None,
                        effect_size: None,
                        metrics: BTreeMap::from([
                            ("observed_proportion".to_string(), observed_proportion),
                            ("expected_proportion".to_string(), *expected_proportion),
                            ("representation_ratio".to_string(), ratio),
                            (
                                "normalized_entropy".to_string(),
                                entropy.unwrap_or(0.0),
                            ),
                        ]),
                    });
                }
            }
        }
    } else {
        for (group, count) in counts {
            let ratio = *count as f64 / max_count;
            if ratio < config.warning_ratio {
                findings.push(Finding {
                    detector: DetectorKind::Representation,
                    grouping: grouping.label.clone(),
                    sensitive_columns: grouping.columns.clone(),
                    target_column: None,
                    group: Some(group.clone()),
                    severity: if ratio < config.critical_ratio {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    message: format!(
                        "group `{group}` is materially smaller than the largest observed group"
                    ),
                    p_value: None,
                    corrected_p_value: None,
                    effect_size: None,
                    metrics: BTreeMap::from([
                        ("row_count".to_string(), *count as f64),
                        ("proportion".to_string(), *count as f64 / total_f),
                        ("representation_ratio".to_string(), ratio),
                        (
                            "normalized_entropy".to_string(),
                            entropy.unwrap_or(0.0),
                        ),
                    ]),
                });
            }
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;

    use tempfile::NamedTempFile;

    use crate::config::{AuditConfig, GroupingMode, ReferenceDistribution};
    use crate::io::csv::{CsvReadOptions, read_csv};

    use super::audit_dataset;

    #[test]
    fn representation_audit_reports_underrepresented_groups() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender,region\nwoman,north\nwoman,south\nwoman,north\nman,south"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let config = AuditConfig::builder().sensitive_column("gender").build();
        let report = audit_dataset(&dataset, &config).expect("audit report");

        assert_eq!(report.group_summaries.len(), 2);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].group.as_deref(), Some("man"));
    }

    #[test]
    fn representation_audit_uses_reference_distribution_when_present() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender\nwoman\nwoman\nwoman\nwoman\nwoman\nwoman\nwoman\nwoman\nwoman\nwoman\nman\nman"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let baseline = ReferenceDistribution::new(BTreeMap::from([
            ("woman".to_string(), 0.5),
            ("man".to_string(), 0.5),
        ]));
        let config = AuditConfig::builder()
            .sensitive_column("gender")
            .reference_distribution("gender", baseline)
            .build();

        let report = audit_dataset(&dataset, &config).expect("audit report");
        assert!(report.findings.iter().any(|finding| finding.p_value.is_some()));
        assert!(report
            .group_summaries
            .iter()
            .all(|summary| summary.expected_proportion.is_some()));
    }

    #[test]
    fn representation_audit_supports_intersectional_grouping() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender,race\nwoman,a\nwoman,a\nman,b\nman,b\nman,b"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let config = AuditConfig::builder()
            .sensitive_columns(["gender", "race"])
            .grouping_mode(GroupingMode::Both)
            .build();

        let report = audit_dataset(&dataset, &config).expect("audit report");
        assert!(report
            .group_summaries
            .iter()
            .any(|summary| summary.grouping == "gender+race"));
        assert!(report
            .detector_runs
            .iter()
            .any(|run| run.grouping == "gender+race"));
    }
}
