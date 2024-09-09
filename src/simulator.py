from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.feature_extractor import sample_feature_combinations
from src.utils import tag_to_normal


class Simulator:
    def __init__(
        self,
        model,
        feat_list: list[str],
        precomputed_features: Optional[pd.DataFrame] = None,
    ):
        """Initialize the simulator

        model: a trained regressor to estimate the performance of a feature combination.
        feat_list (list[str]): ordered list of features.
        precomputed_features (pd.DataFrame): precomputed features for a given dataset. If set, the estimator will be able to estimate
            the number of swaps, and can return feature combinations that actually contain swaps.
        """
        self.model = model
        self.precomputed_df = precomputed_features
        self.feat_list = feat_list
        self.simulated_df = None

    def sample_combinations(
        self,
        n: int = 1000,
        feat_transformer: Optional[callable] = None,
        create_beaker_experiment: Optional[Path] = None,
    ):
        _, combinations = sample_feature_combinations(
            meta_analyzer_n_samples=n, max_number=10
        )

        # Initialize simulation df with 0s, then we fill it with 1s
        # depending on the sampled combinations.
        sim_df = pd.DataFrame(
            0,
            index=np.arange(len(combinations)),
            columns=self.feat_list,
        )
        for idx, combination in tqdm(enumerate(combinations), total=len(combinations)):
            activated_feats = [tag_to_normal(feat) for feat in combination]
            sim_df.loc[idx, activated_feats] = 1

        if feat_transformer:
            sim_df = sim_df.apply(feat_transformer, axis=1)

        sim_df = sim_df.dropna(axis=1).drop_duplicates().reset_index(drop=True)

        # Get predictions
        sim_df["pred"] = self.model.predict(sim_df)
        sim_df = sim_df.sort_values(by="pred", ascending=False).reset_index(drop=True)

        # Precompute
        # TODO

        if create_beaker_experiment:
            # TODO
            pass

        return sim_df
