use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use bias_rs::{
    AuditConfig, CategoricalAssociationConfig, ColumnSelection, DetectorConfig, DetectorKind,
    GroupingMode, MissingnessConfig, MultipleTestingCorrection, NumericDistributionConfig,
    ReferenceDistribution, RepresentationConfig, audit_dataset, read_csv,
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
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum InputFormat {
    Csv,
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
    let dataset = match args.format {
        InputFormat::Csv => read_csv(&args.input, bias_rs::CsvReadOptions::default()),
    }
    .map_err(|error| error.to_string())?;

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

    builder = configure_detectors(builder, &args.detectors);
    if let Some(path) = args.expected_distribution {
        let distributions = load_reference_distributions(&path)?;
        for (grouping, distribution) in distributions {
            builder = builder.reference_distribution(grouping, distribution);
        }
    }

    let config = builder.build();
    let report = audit_dataset(&dataset, &config).map_err(|error| error.to_string())?;

    match args.output {
        OutputFormat::Json => serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize report: {error}")),
        OutputFormat::Text => Ok(render_report(&report)),
    }
}

fn configure_detectors(
    mut builder: bias_rs::AuditConfigBuilder,
    detectors: &[CliDetector],
) -> bias_rs::AuditConfigBuilder {
    if detectors.is_empty() {
        return builder;
    }

    for detector in [
        DetectorKind::Representation,
        DetectorKind::Missingness,
        DetectorKind::CategoricalAssociation,
        DetectorKind::NumericDistribution,
    ] {
        builder = builder.disable_detector(detector);
    }

    for detector in detectors {
        builder = builder.detector(match detector {
            CliDetector::Representation => {
                DetectorConfig::Representation(RepresentationConfig::default())
            }
            CliDetector::Missingness => DetectorConfig::Missingness(MissingnessConfig::default()),
            CliDetector::CategoricalAssociation => {
                DetectorConfig::CategoricalAssociation(CategoricalAssociationConfig::default())
            }
            CliDetector::NumericDistribution => {
                DetectorConfig::NumericDistribution(NumericDistributionConfig::default())
            }
        });
    }

    builder
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
            "- {:?} on {}: {} findings across {} columns\n",
            run.detector, run.grouping, run.finding_count, run.analyzed_columns
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
                "- {:?} on {}{}: {}\n",
                skipped.detector,
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
                "- {:?} {:?}: {}",
                finding.detector, finding.severity, finding.message
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
