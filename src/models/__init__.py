"""
Models package for the NFL OT / 4th-down decision model.

Available sub-modules
---------------------
fg_probability
    Field goal make probability model.
    Entry point: ``python -m src.models.fg_probability``
    Inference:   ``from src.models.fg_probability import predict_fg_prob``
"""

from src.models.fg_probability import (  # noqa: F401
    predict_fg_prob,
    predict_fg_prob_batch,
    build_training_data,
    train,
    save_model,
    load_model,
)
