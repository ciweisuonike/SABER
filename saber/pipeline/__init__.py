# saber/pipeline/__init__.py

"""
Pipeline module for SABER

This module contains the full pipeline for the SABER framework,
which includes the simulation, preprocessing, training, and evaluation stages.
"""
from .simulate import run_simulation
from .preprocess import preprocess_simulated_data
from .train import run_training
from .prop_train import run_prop_train_from_scRNAseq
from .predict import predict

__all__ = [
    "run_simulation",
    "preprocess_simulated_data",
    "run_training",
    "run_prop_train_from_scRNAseq"
]
