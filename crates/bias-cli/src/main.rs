use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use bias_rs::{
    AuditConfig, CategoricalAssociationConfig, ColumnSelection, DetectorConfig, DetectorKind,
    GroupingMode, MissingnessConfig, MultipleTestingCorrection, NumericDistributionConfig,
    ReferenceDistribution, RepresentationConfig, audit_dataset, read_csv, read_parquet,
};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "bias")]
#[command(about = "Audit tabular datasets for group imbalance and distribution shift.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Audit(AuditArgs),
}

#[derive(Debug, Args)]
struct AuditArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, value_enum, default_value_t = InputFormat::Csv)]
    format: InputFormat,
    #[arg(long = "sensitive", required = true)]
    sensitive_columns: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    columns: Vec<String>,
    #[arg(long, value_enum, default_value_t = CliGroupingMode::PerColumn)]
    grouping: CliGroupingMode,
    #[arg(long = "detector", value_enum)]
    detectors: Vec<CliDetector>,
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    #[arg(long = "min-group-size", default_value_t = 30)]
    min_group_size: usize,
    #[arg(long, value_enum, default_value_t = CliCorrection::BenjaminiHochberg)]
    correction: CliCorrection,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,
    #[arg(long = "expected-dist")]
    expected_distribution: Option<PathBuf>,
    #[command(flatten)]
    threshold_args: ThresholdArgs,
}

#[derive(Debug, Args, Clone, Default)]
struct ThresholdArgs {
    #[arg(long = "representation-warning-ratio")]
    representation_warning_ratio: Option<f64>,
    #[arg(long = "representation-critical-ratio")]
    representation_critical_ratio: Option<f64>,
    #[arg(long = "missingness-critical-rate-gap")]
    missingness_critical_rate_gap: Option<f64>,
    #[arg(long = "categorical-critical-cramers-v")]
    categorical_critical_cramers_v: Option<f64>,
    #[arg(long = "numeric-critical-cliffs-delta")]
    numeric_critical_cliffs_delta: Option<f64>,
    #[arg(long = "numeric-critical-epsilon-squared")]
    numeric_critical_epsilon_squared: Option<f64>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliGroupingMode {
    PerColumn,
    Intersectional,
    Both,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliCorrection {
    None,
    BenjaminiHochberg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDetector {
    Representation,
    Missingness,
    CategoricalAssociation,
    NumericDistribution,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<String, String> {
    match cli.command {
        Commands::Audit(args) => run_audit(args),
    }
}

fn run_audit(args: AuditArgs) -> Result<String, String> {
    let detectors = build_detector_configs(&args.detectors, &args.threshold_args)?;
    let mut builder = AuditConfig::builder()
        .sensitive_columns(args.sensitive_columns.clone())
        .grouping_mode(args.grouping.into())
        .alpha(args.alpha)
        .min_group_size(args.min_group_size)
        .multiple_testing(args.correction.into());
    if args.columns.is_empty() {
        builder = builder.analysis_columns(ColumnSelection::All);
    } else {
        builder = builder.analysis_columns(ColumnSelection::Named(args.columns));
    }

    for detector in all_detector_kinds() {
        builder = builder.disable_detector(detector);
    }
    for detector in detectors {
        builder = builder.detector(detector);
    }
    if let Some(path) = args.expected_distribution {
        let distributions = load_reference_distributions(&path)?;
        for (grouping, distribution) in distributions {
            builder = builder.reference_distribution(grouping, distribution);
        }
    }

    let config = builder.build();
    let dataset = match args.format {
        InputFormat::Csv => read_csv(&args.input, bias_rs::CsvReadOptions::default()),
        InputFormat::Parquet => read_parquet(&args.input, bias_rs::ParquetReadOptions::default()),
    }
    .map_err(|error| error.to_string())?;
    let report = audit_dataset(&dataset, &config).map_err(|error| error.to_string())?;

    match args.output {
        OutputFormat::Json => serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize report: {error}")),
        OutputFormat::Text => Ok(render_report(&report)),
    }
}

fn all_detector_kinds() -> [DetectorKind; 4] {
    [
        DetectorKind::Representation,
        DetectorKind::Missingness,
        DetectorKind::CategoricalAssociation,
        DetectorKind::NumericDistribution,
    ]
}

fn build_detector_configs(
    detectors: &[CliDetector],
    threshold_args: &ThresholdArgs,
) -> Result<Vec<DetectorConfig>, String> {
    let selected = selected_detector_kinds(detectors);
    let mut representation = RepresentationConfig::default();
    let mut missingness = MissingnessConfig::default();
    let mut categorical = CategoricalAssociationConfig::default();
    let mut numeric = NumericDistributionConfig::default();

    if let Some(value) = threshold_args.representation_warning_ratio {
        require_selected(
            &selected,
            DetectorKind::Representation,
            "--representation-warning-ratio",
        )?;
        representation.warning_ratio = value;
    }
    if let Some(value) = threshold_args.representation_critical_ratio {
        require_selected(
            &selected,
            DetectorKind::Representation,
            "--representation-critical-ratio",
        )?;
        representation.critical_ratio = value;
    }
    if let Some(value) = threshold_args.missingness_critical_rate_gap {
        require_selected(
            &selected,
            DetectorKind::Missingness,
            "--missingness-critical-rate-gap",
        )?;
        missingness.critical_rate_gap = value;
    }
    if let Some(value) = threshold_args.categorical_critical_cramers_v {
        require_selected(
            &selected,
            DetectorKind::CategoricalAssociation,
            "--categorical-critical-cramers-v",
        )?;
        categorical.critical_cramers_v = value;
    }
    if let Some(value) = threshold_args.numeric_critical_cliffs_delta {
        require_selected(
            &selected,
            DetectorKind::NumericDistribution,
            "--numeric-critical-cliffs-delta",
        )?;
        numeric.critical_cliffs_delta = value;
    }
    if let Some(value) = threshold_args.numeric_critical_epsilon_squared {
        require_selected(
            &selected,
            DetectorKind::NumericDistribution,
            "--numeric-critical-epsilon-squared",
        )?;
        numeric.critical_epsilon_squared = value;
    }

    let mut configs = Vec::new();
    for detector in all_detector_kinds() {
        if !selected.contains(&detector) {
            continue;
        }
        configs.push(match detector {
            DetectorKind::Representation => DetectorConfig::Representation(representation.clone()),
            DetectorKind::Missingness => DetectorConfig::Missingness(missingness.clone()),
            DetectorKind::CategoricalAssociation => {
                DetectorConfig::CategoricalAssociation(categorical.clone())
            }
            DetectorKind::NumericDistribution => {
                DetectorConfig::NumericDistribution(numeric.clone())
            }
        });
    }

    Ok(configs)
}

fn selected_detector_kinds(detectors: &[CliDetector]) -> BTreeSet<DetectorKind> {
    if detectors.is_empty() {
        return all_detector_kinds().into_iter().collect();
    }

    detectors.iter().copied().map(Into::into).collect()
}

fn require_selected(
    selected: &BTreeSet<DetectorKind>,
    detector: DetectorKind,
    flag: &str,
) -> Result<(), String> {
    if selected.contains(&detector) {
        Ok(())
    } else {
        Err(format!(
            "{flag} requires the {} detector",
            detector_name(detector)
        ))
    }
}

fn load_reference_distributions(
    path: &PathBuf,
) -> Result<BTreeMap<String, ReferenceDistribution>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read expected distribution file: {error}"))?;
    let parsed: BTreeMap<String, BTreeMap<String, f64>> = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse expected distribution file: {error}"))?;
    Ok(parsed
        .into_iter()
        .map(|(grouping, groups)| (grouping, ReferenceDistribution::new(groups)))
        .collect())
}

fn render_report(report: &bias_rs::AuditReport) -> String {
    let mut output = String::new();
    output.push_str("Dataset summary\n");
    output.push_str(&format!("rows: {}\n", report.dataset.row_count));
    output.push_str(&format!("columns: {}\n", report.dataset.column_count));
    output.push_str(&format!("batches: {}\n", report.dataset.batch_count));
    output.push('\n');

    output.push_str("Detector runs\n");
    for run in &report.detector_runs {
        output.push_str(&format!(
            "- {} on {}: {} findings across {} columns\n",
            detector_name(run.detector),
            run.grouping,
            run.finding_count,
            run.analyzed_columns
        ));
    }
    output.push('\n');

    if !report.group_summaries.is_empty() {
        output.push_str("Group summaries\n");
        for summary in &report.group_summaries {
            output.push_str(&format!(
                "- {} [{}]: {} rows ({:.2}%)\n",
                summary.grouping,
                summary.group,
                summary.row_count,
                summary.proportion * 100.0
            ));
        }
        output.push('\n');
    }

    if !report.skipped.is_empty() {
        output.push_str("Skipped analyses\n");
        for skipped in &report.skipped {
            output.push_str(&format!(
                "- {} on {}{}: {}\n",
                detector_name(skipped.detector),
                skipped.grouping,
                skipped
                    .target_column
                    .as_ref()
                    .map(|column| format!("/{column}"))
                    .unwrap_or_default(),
                skipped.reason
            ));
        }
        output.push('\n');
    }

    output.push_str("Findings\n");
    if report.findings.is_empty() {
        output.push_str("none\n");
    } else {
        for finding in &report.findings {
            output.push_str(&format!(
                "- {} {}: {}",
                detector_name(finding.detector),
                severity_name(finding.severity),
                finding.message
            ));
            if let Some(column) = &finding.target_column {
                output.push_str(&format!(" [column: {column}]"));
            }
            if let Some(group) = &finding.group {
                output.push_str(&format!(" [group: {group}]"));
            }
            if let Some(p_value) = finding.corrected_p_value.or(finding.p_value) {
                output.push_str(&format!(" [p: {p_value:.4}]"));
            }
            output.push('\n');
        }
    }

    output.trim_end().to_string()
}

fn detector_name(detector: DetectorKind) -> &'static str {
    match detector {
        DetectorKind::Representation => "representation",
        DetectorKind::Missingness => "missingness",
        DetectorKind::CategoricalAssociation => "categorical association",
        DetectorKind::NumericDistribution => "numeric distribution",
    }
}

fn severity_name(severity: bias_rs::Severity) -> &'static str {
    match severity {
        bias_rs::Severity::Info => "info",
        bias_rs::Severity::Warning => "warning",
        bias_rs::Severity::Critical => "critical",
    }
}

impl From<CliGroupingMode> for GroupingMode {
    fn from(value: CliGroupingMode) -> Self {
        match value {
            CliGroupingMode::PerColumn => Self::PerSensitiveColumn,
            CliGroupingMode::Intersectional => Self::Intersectional,
            CliGroupingMode::Both => Self::Both,
        }
    }
}

impl From<CliCorrection> for MultipleTestingCorrection {
    fn from(value: CliCorrection) -> Self {
        match value {
            CliCorrection::None => Self::None,
            CliCorrection::BenjaminiHochberg => Self::BenjaminiHochberg,
        }
    }
}

impl From<CliDetector> for DetectorKind {
    fn from(value: CliDetector) -> Self {
        match value {
            CliDetector::Representation => Self::Representation,
            CliDetector::Missingness => Self::Missingness,
            CliDetector::CategoricalAssociation => Self::CategoricalAssociation,
            CliDetector::NumericDistribution => Self::NumericDistribution,
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Commands, ThresholdArgs, build_detector_configs};

    #[test]
    fn parses_threshold_flags() {
        let cli = Cli::try_parse_from([
            "bias",
            "audit",
            "--input",
            "data.csv",
            "--sensitive",
            "gender",
            "--missingness-critical-rate-gap",
            "0.4",
            "--numeric-critical-epsilon-squared",
            "0.5",
        ])
        .expect("cli");

        let Commands::Audit(args) = cli.command;
        assert_eq!(args.threshold_args.missingness_critical_rate_gap, Some(0.4));
        assert_eq!(
            args.threshold_args.numeric_critical_epsilon_squared,
            Some(0.5)
        );
    }

    #[test]
    fn rejects_threshold_override_for_disabled_detector() {
        let error = build_detector_configs(
            &[super::CliDetector::Representation],
            &ThresholdArgs {
                missingness_critical_rate_gap: Some(0.4),
                ..ThresholdArgs::default()
            },
        )
        .expect_err("disabled detector");

        assert!(error.contains("missingness"));
    }

    #[test]
    fn applies_cli_threshold_overrides() {
        let configs = build_detector_configs(
            &[],
            &ThresholdArgs {
                representation_warning_ratio: Some(0.75),
                numeric_critical_cliffs_delta: Some(0.6),
                ..ThresholdArgs::default()
            },
        )
        .expect("configs");

        let representation = configs
            .iter()
            .find_map(|config| match config {
                bias_rs::DetectorConfig::Representation(settings) => Some(settings),
                _ => None,
            })
            .expect("representation config");
        assert_eq!(representation.warning_ratio, 0.75);

        let numeric = configs
            .iter()
            .find_map(|config| match config {
                bias_rs::DetectorConfig::NumericDistribution(settings) => Some(settings),
                _ => None,
            })
            .expect("numeric config");
        assert_eq!(numeric.critical_cliffs_delta, 0.6);
    }

    #[test]
    fn narrows_configs_to_selected_detectors() {
        let configs = build_detector_configs(
            &[
                super::CliDetector::Missingness,
                super::CliDetector::Representation,
            ],
            &ThresholdArgs::default(),
        )
        .expect("configs");

        assert_eq!(configs.len(), 2);
        assert!(
            configs
                .iter()
                .any(|config| matches!(config, bias_rs::DetectorConfig::Representation(_)))
        );
        assert!(
            configs
                .iter()
                .any(|config| matches!(config, bias_rs::DetectorConfig::Missingness(_)))
        );
        assert!(
            !configs
                .iter()
                .any(|config| matches!(config, bias_rs::DetectorConfig::NumericDistribution(_)))
        );
    }
}
