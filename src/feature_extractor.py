import math
import logging
from typing import Any, Optional

import pandas as pd


class FeatureExtractor:
    """Feature extractor class that takes in a dataframe of prompts, completions, and other metadata

    Each extractor returns a boolean list, indicating if a certain instance fulfills a requirement.
    You can set a `threshold` in the __call__ function to control how many features need to be active.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        prompt_col: str = "text",
        completion_a_col: str = "completion_a",
        completion_b_col: str = "completion_b",
    ):
        self.columns = list(df.columns)
        self.prompts: list[str] = df[prompt_col].to_list()
        self.completion_a: list[str] = df[completion_a_col].to_list()
        self.completion_b: list[str] = df[completion_b_col].to_list()
        logging.info(f"Found {len(self.prompts)} prompts with cols: {self.columns}")

        # Register all extractors here with a shorthand name
        self.REGISTERED_EXTRACTORS = {
            "identity": self._extract_identity,
        }

    def __call__(self, features: list[str], threshold: float = 1.0) -> "pd.DataFrame":
        # boolean matrix of size (n_instances, n_feats)
        result_matrix: list[list[int]] = []
        for feature in features:
            key, params = self.parse_feature(feature)
            if key in self.REGISTERED_EXTRACTORS:
                logging.info(f"Extracting {key} with params: {params}")
                fn = self.REGISTERED_EXTRACTORS[key]
                results = fn(**params)
                result_matrix.append(results)

        # Get all instances that fulfills all (or some) values
        n_features = len(features)
        n_active_to_pass = math.floor(n_features * threshold)
        logging.info(
            f"Getting instances. Needs {n_active_to_pass}/{n_features} to swap."
        )

        # TODO: swap features

    def parse_feature(self, s: str) -> tuple[str, dict[str, Any]]:
        key, params_str = s.split("::")
        params = dict(item.split("=") for item in params_str.split(","))
        params = {k: int(v) if v.isdigit() else v for k, v in params.items()}
        return key, params

    def _extract_identity(self, **kwargs) -> list[int]:
        return [1 for _ in len(self.prompts)]
