from typing import Optional

import pandas as pd
from tqdm import tqdm
import numpy as np

from src.feature_extractor import sample_feature_combinations
from src.utils import tag_to_normal


class Simulator:
    def __init__(
        self,
        model,
        ordered_feat_list: list[str],
        precomputed_features: Optional[pd.DataFrame] = None,
    ):
        """Initialize the simulator

        model: a trained regressor to estimate the performance of a feature combination.
        ordered_feat_list (list[str]): ordered list of features.
        precomputed_features (pd.DataFrame): precomputed features for a given dataset. If set, the estimator will be able to estimate
            the number of swaps, and can return feature combinations that actually contain swaps.
        """
        self.model = model
        self.precomputed_df = precomputed_features
        self.ordered_feat_list = ordered_feat_list

    def sample_combinations(self, n: int = 1000):
        _, combinations = sample_feature_combinations(
            meta_analyzer_n_samples=n, max_number=10
        )

        sim_df = pd.DataFrame(0, index=np.arange())
        for idx, combination in tqdm(enumerate(combinations), total=len(combinations)):
            activated_feats = [tag_to_normal(feat) for feat in combination]
            breakpoint()
