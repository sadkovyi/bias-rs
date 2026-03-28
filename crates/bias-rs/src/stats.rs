use std::cmp::Ordering;

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ChiSquareResult {
    pub statistic: f64,
    pub degrees_of_freedom: usize,
    pub p_value: f64,
    pub min_expected_count: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MannWhitneyResult {
    pub u_statistic: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct KruskalWallisResult {
    pub statistic: f64,
    pub degrees_of_freedom: usize,
    pub p_value: f64,
    pub epsilon_squared: f64,
}

pub(crate) fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    if p_values.is_empty() {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| total_order(left.1, right.1));

    let total = indexed.len() as f64;
    let mut adjusted_sorted = vec![1.0; indexed.len()];
    let mut running_min = 1.0_f64;
    for position in (0..indexed.len()).rev() {
        let rank = (position + 1) as f64;
        let candidate = (indexed[position].1 * total / rank).clamp(0.0, 1.0);
        running_min = running_min.min(candidate);
        adjusted_sorted[position] = running_min;
    }

    let mut adjusted = vec![1.0; indexed.len()];
    for (position, (original_index, _)) in indexed.into_iter().enumerate() {
        adjusted[original_index] = adjusted_sorted[position];
    }

    adjusted
}

pub(crate) fn normalized_entropy(counts: &[usize]) -> Option<f64> {
    if counts.len() < 2 {
        return None;
    }

    let total = counts.iter().sum::<usize>() as f64;
    if total == 0.0 {
        return None;
    }

    let entropy = counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let probability = count as f64 / total;
            -probability * probability.ln()
        })
        .sum::<f64>();
    let scale = (counts.len() as f64).ln();
    if scale == 0.0 {
        None
    } else {
        Some((entropy / scale).clamp(0.0, 1.0))
    }
}

pub(crate) fn chi_square_test(table: &[Vec<u64>]) -> Option<ChiSquareResult> {
    let rows = table.len();
    if rows < 2 {
        return None;
    }
    let cols = table.first()?.len();
    if cols < 2 || table.iter().any(|row| row.len() != cols) {
        return None;
    }

    let row_totals: Vec<f64> = table
        .iter()
        .map(|row| row.iter().sum::<u64>() as f64)
        .collect();
    let mut column_totals = vec![0.0; cols];
    let mut total = 0.0;
    for row in table {
        for (index, value) in row.iter().enumerate() {
            let value = *value as f64;
            column_totals[index] += value;
            total += value;
        }
    }
    if total == 0.0 {
        return None;
    }

    let mut statistic = 0.0;
    let mut min_expected_count = f64::MAX;
    for (row_index, row) in table.iter().enumerate() {
        for (column_index, observed) in row.iter().enumerate() {
            let expected = row_totals[row_index] * column_totals[column_index] / total;
            if expected <= 0.0 {
                return None;
            }
            min_expected_count = min_expected_count.min(expected);
            let delta = *observed as f64 - expected;
            statistic += delta * delta / expected;
        }
    }

    let degrees_of_freedom = (rows - 1) * (cols - 1);
    let distribution = ChiSquared::new(degrees_of_freedom as f64).ok()?;
    let p_value = 1.0 - distribution.cdf(statistic);

    Some(ChiSquareResult {
        statistic,
        degrees_of_freedom,
        p_value,
        min_expected_count,
    })
}

pub(crate) fn cramers_v(table: &[Vec<u64>], chi_square: f64) -> Option<f64> {
    let rows = table.len();
    let cols = table.first()?.len();
    if rows < 2 || cols < 2 {
        return None;
    }

    let total = table
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .sum::<u64>() as f64;
    if total == 0.0 {
        return None;
    }

    let scale = (rows.min(cols) - 1) as f64;
    if scale == 0.0 {
        return None;
    }

    Some((chi_square / (total * scale)).sqrt().clamp(0.0, 1.0))
}

pub(crate) fn goodness_of_fit(
    observed_counts: &[u64],
    expected_proportions: &[f64],
) -> Option<ChiSquareResult> {
    if observed_counts.len() != expected_proportions.len() || observed_counts.len() < 2 {
        return None;
    }

    let total = observed_counts.iter().copied().sum::<u64>() as f64;
    if total == 0.0 {
        return None;
    }

    let mut statistic = 0.0;
    let mut min_expected_count = f64::MAX;
    for (observed, expected_proportion) in observed_counts.iter().zip(expected_proportions.iter()) {
        if *expected_proportion <= 0.0 {
            return None;
        }
        let expected = total * expected_proportion;
        min_expected_count = min_expected_count.min(expected);
        let delta = *observed as f64 - expected;
        statistic += delta * delta / expected;
    }

    let degrees_of_freedom = observed_counts.len() - 1;
    let distribution = ChiSquared::new(degrees_of_freedom as f64).ok()?;
    let p_value = 1.0 - distribution.cdf(statistic);

    Some(ChiSquareResult {
        statistic,
        degrees_of_freedom,
        p_value,
        min_expected_count,
    })
}

pub(crate) fn fisher_exact_2x2(table: [[u64; 2]; 2]) -> Option<f64> {
    let row_one = table[0][0] + table[0][1];
    let row_two = table[1][0] + table[1][1];
    let column_one = table[0][0] + table[1][0];
    let total = row_one + row_two;
    if total == 0 {
        return None;
    }

    let lower = column_one.saturating_sub(row_two);
    let upper = column_one.min(row_one);
    let observed_probability = hypergeometric_probability(total, column_one, row_one, table[0][0]);
    let mut p_value = 0.0;
    for current in lower..=upper {
        let probability = hypergeometric_probability(total, column_one, row_one, current);
        if probability <= observed_probability + 1e-12 {
            p_value += probability;
        }
    }

    Some(p_value.clamp(0.0, 1.0))
}

pub(crate) fn mann_whitney_u(left: &[f64], right: &[f64]) -> Option<MannWhitneyResult> {
    if left.is_empty() || right.is_empty() {
        return None;
    }

    let left_len = left.len() as f64;
    let right_len = right.len() as f64;
    let total_len = left.len() + right.len();

    let mut values = Vec::with_capacity(total_len);
    values.extend(left.iter().copied().map(|value| (value, 0usize)));
    values.extend(right.iter().copied().map(|value| (value, 1usize)));
    values.sort_by(|left, right| total_order(left.0, right.0));

    let (ranks, tie_counts) =
        average_ranks(&values.iter().map(|entry| entry.0).collect::<Vec<_>>());
    let left_rank_sum = values
        .iter()
        .zip(ranks.iter())
        .filter_map(|((_, group), rank)| (*group == 0).then_some(*rank))
        .sum::<f64>();

    let u_left = left_rank_sum - (left_len * (left_len + 1.0) / 2.0);
    let u_right = left_len * right_len - u_left;
    let u_statistic = u_left.min(u_right);

    let mean = left_len * right_len / 2.0;
    let tie_sum = tie_counts
        .iter()
        .map(|count| (*count as f64).powi(3) - *count as f64)
        .sum::<f64>();
    let total_len_f = total_len as f64;
    let variance = (left_len * right_len / 12.0)
        * ((total_len_f + 1.0) - tie_sum / (total_len_f * (total_len_f - 1.0)));
    if variance <= 0.0 {
        return None;
    }

    let z_score = (u_statistic - mean).abs() / variance.sqrt();
    let distribution = Normal::new(0.0, 1.0).ok()?;
    let p_value = 2.0 * (1.0 - distribution.cdf(z_score));

    Some(MannWhitneyResult {
        u_statistic,
        p_value: p_value.clamp(0.0, 1.0),
    })
}

pub(crate) fn cliffs_delta(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.is_empty() || right.is_empty() {
        return None;
    }

    let mut wins = 0.0;
    let mut losses = 0.0;
    for left_value in left {
        for right_value in right {
            if left_value > right_value {
                wins += 1.0;
            } else if left_value < right_value {
                losses += 1.0;
            }
        }
    }

    let total = (left.len() * right.len()) as f64;
    Some(((wins - losses) / total).clamp(-1.0, 1.0))
}

pub(crate) fn kruskal_wallis(groups: &[Vec<f64>]) -> Option<KruskalWallisResult> {
    let non_empty_groups: Vec<_> = groups.iter().filter(|group| !group.is_empty()).collect();
    if non_empty_groups.len() < 2 {
        return None;
    }

    let mut combined = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        combined.extend(group.iter().copied().map(|value| (value, group_index)));
    }
    combined.sort_by(|left, right| total_order(left.0, right.0));

    let (ranks, tie_counts) =
        average_ranks(&combined.iter().map(|entry| entry.0).collect::<Vec<_>>());
    let mut rank_sums = vec![0.0; groups.len()];
    let mut counts = vec![0usize; groups.len()];
    for ((_, group_index), rank) in combined.iter().zip(ranks.iter()) {
        rank_sums[*group_index] += *rank;
        counts[*group_index] += 1;
    }

    let total = combined.len() as f64;
    let mut statistic = 0.0;
    for (rank_sum, count) in rank_sums.iter().zip(counts.iter()) {
        if *count > 0 {
            statistic += rank_sum * rank_sum / *count as f64;
        }
    }
    statistic = (12.0 / (total * (total + 1.0))) * statistic - 3.0 * (total + 1.0);

    let tie_sum = tie_counts
        .iter()
        .map(|count| (*count as f64).powi(3) - *count as f64)
        .sum::<f64>();
    let correction = 1.0 - tie_sum / (total.powi(3) - total);
    if correction <= 0.0 {
        return None;
    }
    statistic /= correction;

    let degrees_of_freedom = non_empty_groups.len() - 1;
    let distribution = ChiSquared::new(degrees_of_freedom as f64).ok()?;
    let p_value = 1.0 - distribution.cdf(statistic);
    let epsilon_squared = if total > non_empty_groups.len() as f64 {
        ((statistic - degrees_of_freedom as f64) / (total - non_empty_groups.len() as f64))
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    Some(KruskalWallisResult {
        statistic,
        degrees_of_freedom,
        p_value,
        epsilon_squared,
    })
}

fn average_ranks(values: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut ranks = vec![0.0; values.len()];
    let mut tie_counts = Vec::new();
    let mut start = 0usize;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && total_order(values[start], values[end]) == Ordering::Equal {
            end += 1;
        }

        let average_rank = (start + 1 + end) as f64 / 2.0;
        for rank in &mut ranks[start..end] {
            *rank = average_rank;
        }
        if end - start > 1 {
            tie_counts.push(end - start);
        }
        start = end;
    }

    (ranks, tie_counts)
}

fn hypergeometric_probability(total: u64, successes: u64, draws: u64, observed: u64) -> f64 {
    let failures = total - successes;
    let observed_failures = draws - observed;
    let log_probability = ln_choose(successes, observed) + ln_choose(failures, observed_failures)
        - ln_choose(total, draws);
    log_probability.exp()
}

fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

fn total_order(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::{
        benjamini_hochberg, chi_square_test, cliffs_delta, fisher_exact_2x2, goodness_of_fit,
        kruskal_wallis, mann_whitney_u, normalized_entropy,
    };

    #[test]
    fn benjamini_hochberg_preserves_order() {
        let adjusted = benjamini_hochberg(&[0.01, 0.04, 0.03]);
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0] <= adjusted[1]);
        assert!(adjusted[0] <= adjusted[2]);
    }

    #[test]
    fn entropy_is_scaled() {
        let entropy = normalized_entropy(&[50, 50]).expect("entropy");
        assert!((entropy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chi_square_detects_association() {
        let result = chi_square_test(&[vec![30, 10], vec![10, 30]]).expect("chi-square");
        assert!(result.p_value < 0.01);
        assert_eq!(result.degrees_of_freedom, 1);
    }

    #[test]
    fn fisher_exact_handles_sparse_tables() {
        let p_value = fisher_exact_2x2([[1, 9], [8, 2]]).expect("fisher");
        assert!(p_value < 0.05);
    }

    #[test]
    fn goodness_of_fit_flags_large_shift() {
        let result = goodness_of_fit(&[70, 30], &[0.5, 0.5]).expect("gof");
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn rank_tests_work_on_shifted_data() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [8.0, 9.0, 10.0, 11.0];
        let mann_whitney = mann_whitney_u(&left, &right).expect("mann-whitney");
        let delta = cliffs_delta(&left, &right).expect("delta");
        assert!(mann_whitney.p_value < 0.05);
        assert!(delta < -0.9);
    }

    #[test]
    fn kruskal_wallis_detects_group_shift() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![5.0, 6.0, 7.0],
            vec![8.0, 9.0, 10.0],
        ];
        let result = kruskal_wallis(&groups).expect("kruskal-wallis");
        assert!(result.p_value < 0.05);
        assert!(result.epsilon_squared > 0.5);
    }
}
