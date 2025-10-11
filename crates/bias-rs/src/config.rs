use std::collections::BTreeMap;

/// Controls which columns are analyzed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnSelection {
    All,
    Named(Vec<String>),
}

/// Controls how sensitive columns are grouped during an audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupingMode {
    PerSensitiveColumn,
    Intersectional,
    Both,
}

/// Supported multiple testing correction strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MultipleTestingCorrection {
    None,
    BenjaminiHochberg,
}

/// Distinguishes the supported detector types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorKind {
    Representation,
    Missingness,
    CategoricalAssociation,
    NumericDistribution,
}

/// Configuration for representation analysis.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RepresentationConfig {
    pub warning_ratio: f64,
    pub critical_ratio: f64,
}

impl Default for RepresentationConfig {
    fn default() -> Self {
        Self {
            warning_ratio: 0.8,
            critical_ratio: 0.5,
        }
    }
}

/// Configuration for missingness analysis.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MissingnessConfig {
    pub sparse_table_threshold: usize,
}

impl Default for MissingnessConfig {
    fn default() -> Self {
        Self {
            sparse_table_threshold: 5,
        }
    }
}

/// Configuration for categorical association analysis.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CategoricalAssociationConfig {
    pub max_categories: usize,
    pub rare_category_threshold: usize,
}

impl Default for CategoricalAssociationConfig {
    fn default() -> Self {
        Self {
            max_categories: 32,
            rare_category_threshold: 5,
        }
    }
}

/// Configuration for numeric distribution analysis.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct NumericDistributionConfig {
    pub drop_missing: bool,
}

impl Default for NumericDistributionConfig {
    fn default() -> Self {
        Self { drop_missing: true }
    }
}

/// User-configurable detector settings.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DetectorConfig {
    Representation(RepresentationConfig),
    Missingness(MissingnessConfig),
    CategoricalAssociation(CategoricalAssociationConfig),
    NumericDistribution(NumericDistributionConfig),
}

impl DetectorConfig {
    /// Returns the kind for a detector configuration.
    pub fn kind(&self) -> DetectorKind {
        match self {
            Self::Representation(_) => DetectorKind::Representation,
            Self::Missingness(_) => DetectorKind::Missingness,
            Self::CategoricalAssociation(_) => DetectorKind::CategoricalAssociation,
            Self::NumericDistribution(_) => DetectorKind::NumericDistribution,
        }
    }
}

/// Expected group proportions for representation analysis.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ReferenceDistribution {
    pub groups: BTreeMap<String, f64>,
}

impl ReferenceDistribution {
    /// Creates a new reference distribution from named proportions.
    pub fn new(groups: BTreeMap<String, f64>) -> Self {
        Self { groups }
    }
}

/// Top-level audit configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AuditConfig {
    pub sensitive_columns: Vec<String>,
    pub analysis_columns: ColumnSelection,
    pub grouping_mode: GroupingMode,
    pub detectors: Vec<DetectorConfig>,
    pub alpha: f64,
    pub multiple_testing: MultipleTestingCorrection,
    pub min_group_size: usize,
    pub reference_distributions: BTreeMap<String, ReferenceDistribution>,
}

impl AuditConfig {
    /// Creates a builder with practical defaults.
    pub fn builder() -> AuditConfigBuilder {
        AuditConfigBuilder::default()
    }
}

/// Builder for audit configuration.
#[derive(Debug, Clone)]
pub struct AuditConfigBuilder {
    sensitive_columns: Vec<String>,
    analysis_columns: ColumnSelection,
    grouping_mode: GroupingMode,
    detectors: BTreeMap<DetectorKind, DetectorConfig>,
    alpha: f64,
    multiple_testing: MultipleTestingCorrection,
    min_group_size: usize,
    reference_distributions: BTreeMap<String, ReferenceDistribution>,
}

impl Default for AuditConfigBuilder {
    fn default() -> Self {
        let mut detectors = BTreeMap::new();
        detectors.insert(
            DetectorKind::Representation,
            DetectorConfig::Representation(RepresentationConfig::default()),
        );
        detectors.insert(
            DetectorKind::Missingness,
            DetectorConfig::Missingness(MissingnessConfig::default()),
        );
        detectors.insert(
            DetectorKind::CategoricalAssociation,
            DetectorConfig::CategoricalAssociation(CategoricalAssociationConfig::default()),
        );
        detectors.insert(
            DetectorKind::NumericDistribution,
            DetectorConfig::NumericDistribution(NumericDistributionConfig::default()),
        );

        Self {
            sensitive_columns: Vec::new(),
            analysis_columns: ColumnSelection::All,
            grouping_mode: GroupingMode::PerSensitiveColumn,
            detectors,
            alpha: 0.05,
            multiple_testing: MultipleTestingCorrection::BenjaminiHochberg,
            min_group_size: 30,
            reference_distributions: BTreeMap::new(),
        }
    }
}

impl AuditConfigBuilder {
    /// Adds a sensitive column.
    pub fn sensitive_column(mut self, name: impl Into<String>) -> Self {
        self.sensitive_columns.push(name.into());
        self
    }

    /// Replaces the full sensitive column list.
    pub fn sensitive_columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sensitive_columns = columns.into_iter().map(Into::into).collect();
        self
    }

    /// Sets which non-sensitive columns are analyzed.
    pub fn analysis_columns(mut self, selection: ColumnSelection) -> Self {
        self.analysis_columns = selection;
        self
    }

    /// Sets the grouping mode.
    pub fn grouping_mode(mut self, mode: GroupingMode) -> Self {
        self.grouping_mode = mode;
        self
    }

    /// Enables or overrides a detector configuration.
    pub fn detector(mut self, detector: DetectorConfig) -> Self {
        self.detectors.insert(detector.kind(), detector);
        self
    }

    /// Disables a detector.
    pub fn disable_detector(mut self, kind: DetectorKind) -> Self {
        self.detectors.remove(&kind);
        self
    }

    /// Sets the significance threshold.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the multiple testing strategy.
    pub fn multiple_testing(mut self, correction: MultipleTestingCorrection) -> Self {
        self.multiple_testing = correction;
        self
    }

    /// Sets the minimum group size for statistical testing.
    pub fn min_group_size(mut self, min_group_size: usize) -> Self {
        self.min_group_size = min_group_size;
        self
    }

    /// Registers an expected distribution for a grouping label.
    pub fn reference_distribution(
        mut self,
        grouping: impl Into<String>,
        distribution: ReferenceDistribution,
    ) -> Self {
        self.reference_distributions
            .insert(grouping.into(), distribution);
        self
    }

    /// Builds the configuration.
    pub fn build(self) -> AuditConfig {
        AuditConfig {
            sensitive_columns: self.sensitive_columns,
            analysis_columns: self.analysis_columns,
            grouping_mode: self.grouping_mode,
            detectors: self.detectors.into_values().collect(),
            alpha: self.alpha,
            multiple_testing: self.multiple_testing,
            min_group_size: self.min_group_size,
            reference_distributions: self.reference_distributions,
        }
    }
}
