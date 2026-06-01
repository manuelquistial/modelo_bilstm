"""Alias: Eff-KNN is the KNN baseline cited in Sun et al. Table 2."""

from src.baselines.eff_knn import EffKNNClassifier

KNNClassifier = EffKNNClassifier

__all__ = ["KNNClassifier", "EffKNNClassifier"]
