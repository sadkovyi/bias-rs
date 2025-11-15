use std::collections::BTreeMap;

use crate::config::{
    CategoricalAssociationConfig, DetectorConfig, DetectorKind, MissingnessConfig,
    NumericDistributionConfig, RepresentationConfig,
};
use crate::report::{AuditReportConfig, DetectorRun, GroupSummary};
use crate::stats::{
    chi_square_test, cliffs_delta, cramers_v, fisher_exact_2x2, goodness_of_fit, kruskal_wallis,
    mann_whitney_u, normalized_entropy,
};
use crate::table::{
    GroupingSpec, analysis_columns, build_group_keys, collect_null_flags, collect_numeric_column,
    collect_stringish_column, grouping_specs, is_numeric_type,
};
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
    let missingness_config = config.detectors.iter().find_map(|detector| match detector {
        DetectorConfig::Missingness(settings) => Some(settings.clone()),
        _ => None,
    });
    let categorical_config = config.detectors.iter().find_map(|detector| match detector {
        DetectorConfig::CategoricalAssociation(settings) => Some(settings.clone()),
        _ => None,
    });
    let numeric_config = config.detectors.iter().find_map(|detector| match detector {
        DetectorConfig::NumericDistribution(settings) => Some(settings.clone()),
        _ => None,
    });
    let analysis_columns = analysis_columns(dataset, config)?;

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

        if let Some(settings) = &missingness_config {
            let mut missingness_findings = missingness_findings(
                dataset,
                grouping,
                &group_keys,
                &counts,
                &analysis_columns,
                settings,
                config.alpha,
                config.min_group_size,
                &mut skipped,
            )?;
            findings.append(&mut missingness_findings);
            detector_runs.push(DetectorRun {
                detector: DetectorKind::Missingness,
                grouping: grouping.label.clone(),
                analyzed_columns: analysis_columns.len(),
                finding_count: findings
                    .iter()
                    .filter(|finding| {
                        finding.detector == DetectorKind::Missingness
                            && finding.grouping == grouping.label
                    })
                    .count(),
            });
        }

        if let Some(settings) = &categorical_config {
            let mut categorical_findings = categorical_association_findings(
                dataset,
                grouping,
                &group_keys,
                &analysis_columns,
                settings,
                config.alpha,
                config.min_group_size,
                &mut skipped,
            )?;
            findings.append(&mut categorical_findings);
            detector_runs.push(DetectorRun {
                detector: DetectorKind::CategoricalAssociation,
                grouping: grouping.label.clone(),
                analyzed_columns: analysis_columns.len(),
                finding_count: findings
                    .iter()
                    .filter(|finding| {
                        finding.detector == DetectorKind::CategoricalAssociation
                            && finding.grouping == grouping.label
                    })
                    .count(),
            });
        }

        if let Some(settings) = &numeric_config {
            let mut numeric_findings = numeric_distribution_findings(
                dataset,
                grouping,
                &group_keys,
                &analysis_columns,
                settings,
                config.alpha,
                config.min_group_size,
                &mut skipped,
            )?;
            findings.append(&mut numeric_findings);
            detector_runs.push(DetectorRun {
                detector: DetectorKind::NumericDistribution,
                grouping: grouping.label.clone(),
                analyzed_columns: analysis_columns.len(),
                finding_count: findings
                    .iter()
                    .filter(|finding| {
                        finding.detector == DetectorKind::NumericDistribution
                            && finding.grouping == grouping.label
                    })
                    .count(),
            });
        }
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

fn missingness_findings(
    dataset: &Dataset,
    grouping: &GroupingSpec,
    group_keys: &[String],
    group_counts: &BTreeMap<String, usize>,
    analysis_columns: &[String],
    config: &MissingnessConfig,
    alpha: f64,
    min_group_size: usize,
    skipped: &mut Vec<SkippedAnalysis>,
) -> Result<Vec<Finding>, BiasError> {
    if group_counts.len() < 2 {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::Missingness,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: "missingness analysis needs at least two groups".to_string(),
        });
        return Ok(Vec::new());
    }

    if group_counts.values().any(|count| *count < min_group_size) {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::Missingness,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: format!("at least one group is smaller than {min_group_size} rows"),
        });
        return Ok(Vec::new());
    }

    let group_order = group_counts.keys().cloned().collect::<Vec<_>>();
    let group_index = group_order
        .iter()
        .enumerate()
        .map(|(index, group)| (group.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut findings = Vec::new();
    for column in analysis_columns {
        let null_flags = collect_null_flags(dataset, column)?;
        let mut table = vec![vec![0_u64, 0_u64]; group_order.len()];
        for (group_key, is_missing) in group_keys.iter().zip(null_flags.iter()) {
            let row = group_index[group_key];
            if *is_missing {
                table[row][0] += 1;
            } else {
                table[row][1] += 1;
            }
        }

        let missing_rates = group_order
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let total = table[index][0] + table[index][1];
                let rate = if total == 0 {
                    0.0
                } else {
                    table[index][0] as f64 / total as f64
                };
                (group.clone(), rate)
            })
            .collect::<Vec<_>>();
        let max_missing_rate = missing_rates
            .iter()
            .map(|(_, rate)| *rate)
            .fold(0.0_f64, f64::max);
        let min_missing_rate = missing_rates
            .iter()
            .map(|(_, rate)| *rate)
            .fold(1.0_f64, f64::min);
        let rate_gap = max_missing_rate - min_missing_rate;

        let chi_square = chi_square_test(&table);
        let fisher_p_value = if table.len() == 2
            && table.iter().flat_map(|row| row.iter()).any(|value| *value < config.sparse_table_threshold as u64)
        {
            fisher_exact_2x2([[table[0][0], table[0][1]], [table[1][0], table[1][1]]])
        } else {
            None
        };

        let (p_value, effect_size, min_expected_count) = if let Some(p_value) = fisher_p_value {
            (Some(p_value), Some(rate_gap), None)
        } else if let Some(result) = chi_square {
            (
                Some(result.p_value),
                cramers_v(&table, result.statistic),
                Some(result.min_expected_count),
            )
        } else {
            skipped.push(SkippedAnalysis {
                detector: DetectorKind::Missingness,
                grouping: grouping.label.clone(),
                target_column: Some(column.clone()),
                reason: "missingness contingency table could not be evaluated".to_string(),
            });
            continue;
        };

        if let Some(p_value) = p_value {
            if p_value <= alpha {
                findings.push(Finding {
                    detector: DetectorKind::Missingness,
                    grouping: grouping.label.clone(),
                    sensitive_columns: grouping.columns.clone(),
                    target_column: Some(column.clone()),
                    group: None,
                    severity: if rate_gap >= 0.25 {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    message: format!("missing values for `{column}` vary across sensitive groups"),
                    p_value: Some(p_value),
                    corrected_p_value: None,
                    effect_size,
                    metrics: BTreeMap::from([
                        ("max_missing_rate".to_string(), max_missing_rate),
                        ("min_missing_rate".to_string(), min_missing_rate),
                        ("missing_rate_gap".to_string(), rate_gap),
                        (
                            "min_expected_count".to_string(),
                            min_expected_count.unwrap_or(0.0),
                        ),
                    ]),
                });
            }
        }
    }

    Ok(findings)
}

fn categorical_association_findings(
    dataset: &Dataset,
    grouping: &GroupingSpec,
    group_keys: &[String],
    analysis_columns: &[String],
    config: &CategoricalAssociationConfig,
    alpha: f64,
    min_group_size: usize,
    skipped: &mut Vec<SkippedAnalysis>,
) -> Result<Vec<Finding>, BiasError> {
    let group_counts = count_groups(group_keys);
    if group_counts.len() < 2 {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::CategoricalAssociation,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: "categorical association needs at least two groups".to_string(),
        });
        return Ok(Vec::new());
    }

    if group_counts.values().any(|count| *count < min_group_size) {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::CategoricalAssociation,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: format!("at least one group is smaller than {min_group_size} rows"),
        });
        return Ok(Vec::new());
    }

    let group_order = group_counts.keys().cloned().collect::<Vec<_>>();
    let group_index = group_order
        .iter()
        .enumerate()
        .map(|(index, group)| (group.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut findings = Vec::new();

    for column in analysis_columns {
        let data_type = dataset
            .column_type(column)
            .ok_or_else(|| BiasError::MissingColumn(column.clone()))?;
        if is_numeric_type(data_type) {
            continue;
        }

        let values = collect_stringish_column(dataset, column)?;
        let categories = collapse_categories(values, config);
        let category_order = categories
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if category_order.len() < 2 {
            continue;
        }

        let category_index = category_order
            .iter()
            .enumerate()
            .map(|(index, category)| (category.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let mut table = vec![vec![0_u64; category_order.len()]; group_order.len()];
        for ((group_key, category), _) in group_keys.iter().zip(categories.iter()).zip(0..) {
            let row = group_index[group_key];
            let column_index = category_index[category];
            table[row][column_index] += 1;
        }

        let Some(chi_square) = chi_square_test(&table) else {
            skipped.push(SkippedAnalysis {
                detector: DetectorKind::CategoricalAssociation,
                grouping: grouping.label.clone(),
                target_column: Some(column.clone()),
                reason: "categorical contingency table could not be evaluated".to_string(),
            });
            continue;
        };

        if chi_square.p_value <= alpha {
            findings.push(Finding {
                detector: DetectorKind::CategoricalAssociation,
                grouping: grouping.label.clone(),
                sensitive_columns: grouping.columns.clone(),
                target_column: Some(column.clone()),
                group: None,
                severity: if cramers_v(&table, chi_square.statistic).unwrap_or(0.0) >= 0.3 {
                    Severity::Critical
                } else {
                    Severity::Warning
                },
                message: format!(
                    "categorical values for `{column}` are distributed differently across sensitive groups"
                ),
                p_value: Some(chi_square.p_value),
                corrected_p_value: None,
                effect_size: cramers_v(&table, chi_square.statistic),
                metrics: BTreeMap::from([
                    ("chi_square".to_string(), chi_square.statistic),
                    (
                        "degrees_of_freedom".to_string(),
                        chi_square.degrees_of_freedom as f64,
                    ),
                    (
                        "min_expected_count".to_string(),
                        chi_square.min_expected_count,
                    ),
                    ("category_count".to_string(), category_order.len() as f64),
                ]),
            });
        }
    }

    Ok(findings)
}

fn collapse_categories(
    values: Vec<Option<String>>,
    config: &CategoricalAssociationConfig,
) -> Vec<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    let normalized = values
        .into_iter()
        .map(|value| value.unwrap_or_else(|| "<missing>".to_string()))
        .collect::<Vec<_>>();
    for value in &normalized {
        *counts.entry(value.clone()).or_insert(0) += 1;
    }

    let mut retained = counts
        .iter()
        .filter_map(|(category, count)| {
            (*count >= config.rare_category_threshold).then_some((category.clone(), *count))
        })
        .collect::<Vec<_>>();
    retained.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    if retained.len() >= config.max_categories {
        retained.truncate(config.max_categories.saturating_sub(1));
    }
    let retained = retained
        .into_iter()
        .map(|(category, _)| category)
        .collect::<std::collections::BTreeSet<_>>();

    normalized
        .into_iter()
        .map(|value| {
            if retained.contains(&value) {
                value
            } else {
                "__OTHER__".to_string()
            }
        })
        .collect()
}

fn numeric_distribution_findings(
    dataset: &Dataset,
    grouping: &GroupingSpec,
    group_keys: &[String],
    analysis_columns: &[String],
    config: &NumericDistributionConfig,
    alpha: f64,
    min_group_size: usize,
    skipped: &mut Vec<SkippedAnalysis>,
) -> Result<Vec<Finding>, BiasError> {
    let group_counts = count_groups(group_keys);
    if group_counts.len() < 2 {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::NumericDistribution,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: "numeric distribution needs at least two groups".to_string(),
        });
        return Ok(Vec::new());
    }

    if group_counts.values().any(|count| *count < min_group_size) {
        skipped.push(SkippedAnalysis {
            detector: DetectorKind::NumericDistribution,
            grouping: grouping.label.clone(),
            target_column: None,
            reason: format!("at least one group is smaller than {min_group_size} rows"),
        });
        return Ok(Vec::new());
    }

    let group_order = group_counts.keys().cloned().collect::<Vec<_>>();
    let group_index = group_order
        .iter()
        .enumerate()
        .map(|(index, group)| (group.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut findings = Vec::new();

    for column in analysis_columns {
        let data_type = dataset
            .column_type(column)
            .ok_or_else(|| BiasError::MissingColumn(column.clone()))?;
        if !is_numeric_type(data_type) {
            continue;
        }

        let values = collect_numeric_column(dataset, column)?;
        let mut grouped_values = vec![Vec::<f64>::new(); group_order.len()];
        for (group_key, value) in group_keys.iter().zip(values.iter()) {
            if let Some(value) = value {
                grouped_values[group_index[group_key]].push(*value);
            } else if !config.drop_missing {
                grouped_values[group_index[group_key]].push(f64::NAN);
            }
        }
        if grouped_values.iter().filter(|values| !values.is_empty()).count() < 2 {
            skipped.push(SkippedAnalysis {
                detector: DetectorKind::NumericDistribution,
                grouping: grouping.label.clone(),
                target_column: Some(column.clone()),
                reason: "numeric values were not present in at least two groups".to_string(),
            });
            continue;
        }

        if grouped_values.len() == 2 {
            let left = grouped_values[0]
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let right = grouped_values[1]
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            let Some(test) = mann_whitney_u(&left, &right) else {
                skipped.push(SkippedAnalysis {
                    detector: DetectorKind::NumericDistribution,
                    grouping: grouping.label.clone(),
                    target_column: Some(column.clone()),
                    reason: "mann-whitney test could not be evaluated".to_string(),
                });
                continue;
            };
            if test.p_value <= alpha {
                let delta = cliffs_delta(&left, &right).unwrap_or(0.0);
                let left_median = median(&left);
                let right_median = median(&right);
                findings.push(Finding {
                    detector: DetectorKind::NumericDistribution,
                    grouping: grouping.label.clone(),
                    sensitive_columns: grouping.columns.clone(),
                    target_column: Some(column.clone()),
                    group: None,
                    severity: if delta.abs() >= 0.33 {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    message: format!(
                        "numeric values for `{column}` differ across sensitive groups"
                    ),
                    p_value: Some(test.p_value),
                    corrected_p_value: None,
                    effect_size: Some(delta),
                    metrics: BTreeMap::from([
                        ("u_statistic".to_string(), test.u_statistic),
                        ("left_median".to_string(), left_median),
                        ("right_median".to_string(), right_median),
                    ]),
                });
            }
        } else {
            let finite_groups = grouped_values
                .iter()
                .map(|values| {
                    values
                        .iter()
                        .copied()
                        .filter(|value| value.is_finite())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let Some(test) = kruskal_wallis(&finite_groups) else {
                skipped.push(SkippedAnalysis {
                    detector: DetectorKind::NumericDistribution,
                    grouping: grouping.label.clone(),
                    target_column: Some(column.clone()),
                    reason: "kruskal-wallis test could not be evaluated".to_string(),
                });
                continue;
            };
            if test.p_value <= alpha {
                findings.push(Finding {
                    detector: DetectorKind::NumericDistribution,
                    grouping: grouping.label.clone(),
                    sensitive_columns: grouping.columns.clone(),
                    target_column: Some(column.clone()),
                    group: None,
                    severity: if test.epsilon_squared >= 0.26 {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    message: format!(
                        "numeric values for `{column}` differ across sensitive groups"
                    ),
                    p_value: Some(test.p_value),
                    corrected_p_value: None,
                    effect_size: Some(test.epsilon_squared),
                    metrics: BTreeMap::from([
                        ("kruskal_wallis_h".to_string(), test.statistic),
                        (
                            "degrees_of_freedom".to_string(),
                            test.degrees_of_freedom as f64,
                        ),
                        ("epsilon_squared".to_string(), test.epsilon_squared),
                    ]),
                });
            }
        }
    }

    Ok(findings)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
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
    use crate::DetectorKind;

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

    #[test]
    fn missingness_audit_flags_group_differences() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender,score\nwoman,\nwoman,\nwoman,\nwoman,\nwoman,\nwoman,\nwoman,\nwoman,1.0\nman,2.0\nman,2.0\nman,3.0\nman,3.0\nman,4.0\nman,4.0\nman,5.0\nman,5.0"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let config = AuditConfig::builder()
            .sensitive_column("gender")
            .min_group_size(2)
            .build();

        let report = audit_dataset(&dataset, &config).expect("audit report");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.detector == DetectorKind::Missingness));
    }

    #[test]
    fn categorical_audit_flags_distribution_shift() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender,region\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,north\nwoman,south\nwoman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,south\nman,north\nman,north"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let config = AuditConfig::builder()
            .sensitive_column("gender")
            .min_group_size(2)
            .build();

        let report = audit_dataset(&dataset, &config).expect("audit report");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.detector == DetectorKind::CategoricalAssociation));
    }

    #[test]
    fn numeric_audit_flags_rank_shift() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(
            file,
            "gender,age\nwoman,22\nwoman,23\nwoman,24\nwoman,25\nwoman,26\nman,40\nman,41\nman,42\nman,43\nman,44"
        )
        .expect("write csv");

        let dataset = read_csv(file.path(), CsvReadOptions::default()).expect("dataset");
        let config = AuditConfig::builder()
            .sensitive_column("gender")
            .min_group_size(2)
            .build();

        let report = audit_dataset(&dataset, &config).expect("audit report");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.detector == DetectorKind::NumericDistribution));
    }
}
