//! Multiclass evaluation: confusion matrix, per-class precision/recall/F1,
//! overall accuracy, macro-F1 and support-weighted F1. Every metric is derived
//! from a single confusion matrix so they stay mutually consistent.

use serde::Serialize;

/// Per-class precision / recall / F1 and the support (true count) for one class.
#[derive(Debug, Clone, Serialize)]
pub struct ClassMetrics {
    pub class_id: usize,
    pub class_name: String,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    /// Number of test clips whose true label is this class.
    pub support: usize,
}

/// The full multiclass evaluation bundle, serialized into the results JSON.
#[derive(Debug, Clone, Serialize)]
pub struct MulticlassReport {
    pub n_classes: usize,
    pub n_test: usize,
    /// `class_names[id]` - the id <-> brand-name map.
    pub class_names: Vec<String>,
    /// `confusion[true][pred]` counts.
    pub confusion: Vec<Vec<usize>>,
    pub per_class: Vec<ClassMetrics>,
    pub accuracy: f32,
    /// Unweighted mean of the per-class F1 scores.
    pub macro_f1: f32,
    /// Support-weighted mean of the per-class F1 scores.
    pub weighted_f1: f32,
}

/// Build the full report from paired `(true_label, predicted_label)` ids.
pub fn evaluate(
    pairs: &[(usize, usize)],
    n_classes: usize,
    class_names: &[String],
) -> MulticlassReport {
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    for &(y, p) in pairs {
        confusion[y][p] += 1;
    }

    let n_test = pairs.len();
    let correct: usize = (0..n_classes).map(|c| confusion[c][c]).sum();
    let accuracy = safe_div(correct as f32, n_test as f32);

    let mut per_class = Vec::with_capacity(n_classes);
    for (c, row) in confusion.iter().enumerate() {
        let tp = row[c];
        let pred_c: usize = confusion.iter().map(|r| r[c]).sum();
        let true_c: usize = row.iter().sum();
        let fp = pred_c - tp;
        let fn_ = true_c - tp;
        let precision = safe_div(tp as f32, (tp + fp) as f32);
        let recall = safe_div(tp as f32, (tp + fn_) as f32);
        let f1 = safe_div(2.0 * precision * recall, precision + recall);
        per_class.push(ClassMetrics {
            class_id: c,
            class_name: class_names.get(c).cloned().unwrap_or_else(|| c.to_string()),
            precision,
            recall,
            f1,
            support: true_c,
        });
    }

    let macro_f1 = if n_classes == 0 {
        0.0
    } else {
        per_class.iter().map(|m| m.f1).sum::<f32>() / n_classes as f32
    };
    let total_support: usize = per_class.iter().map(|m| m.support).sum();
    let weighted_f1 = if total_support == 0 {
        0.0
    } else {
        per_class
            .iter()
            .map(|m| m.f1 * m.support as f32)
            .sum::<f32>()
            / total_support as f32
    };

    MulticlassReport {
        n_classes,
        n_test,
        class_names: class_names.to_vec(),
        confusion,
        per_class,
        accuracy,
        macro_f1,
        weighted_f1,
    }
}

#[inline]
fn safe_div(a: f32, b: f32) -> f32 {
    if b.abs() > f32::EPSILON {
        a / b
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("c{i}")).collect()
    }

    #[test]
    fn perfect_predictions_score_one() {
        let pairs = vec![(0, 0), (1, 1), (2, 2), (0, 0)];
        let r = evaluate(&pairs, 3, &names(3));
        assert!((r.accuracy - 1.0).abs() < 1e-6);
        assert!((r.macro_f1 - 1.0).abs() < 1e-6);
        assert_eq!(r.confusion[0][0], 2);
    }

    #[test]
    fn confusion_and_per_class_are_consistent() {
        // class 0: 2 right; class 1: one misclassified as 0.
        let pairs = vec![(0, 0), (0, 0), (1, 0), (1, 1)];
        let r = evaluate(&pairs, 2, &names(2));
        assert!((r.per_class[0].precision - 2.0 / 3.0).abs() < 1e-5);
        assert!((r.per_class[0].recall - 1.0).abs() < 1e-6);
        assert!((r.per_class[1].recall - 0.5).abs() < 1e-6);
        assert_eq!(r.per_class[1].support, 2);
        let mean = (r.per_class[0].f1 + r.per_class[1].f1) / 2.0;
        assert!((r.macro_f1 - mean).abs() < 1e-6);
    }
}
