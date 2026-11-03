"""
Evaluation package.
"""

from src.evaluation.evaluate import (
    evaluate_test_set,
    plot_training_curves,
    generate_experiment_comparison_table,
)
from src.evaluation.gradcam import (
    compute_gradcam_heatmap,
    overlay_heatmap_on_image,
    explain_predictions_with_gradcam,
)
from src.evaluation.model_comparator import (
    UniversalModelWrapper,
    load_universal_model,
    evaluate_model_on_dataset,
    compare_multiple_models,
    visualize_9_predictions,
    plot_model_comparison_charts,
    enable_legacy_keras2_compatibility,
)

__all__ = [
    "evaluate_test_set",
    "plot_training_curves",
    "generate_experiment_comparison_table",
    "compute_gradcam_heatmap",
    "overlay_heatmap_on_image",
    "explain_predictions_with_gradcam",
    "UniversalModelWrapper",
    "load_universal_model",
    "evaluate_model_on_dataset",
    "compare_multiple_models",
    "visualize_9_predictions",
    "plot_model_comparison_charts",
    "enable_legacy_keras2_compatibility",
]

