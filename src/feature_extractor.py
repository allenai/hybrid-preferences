import math
import logging
from pathlib import Path
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
        *,
        prompt_col: str = "text",
        completion_a_col: str = "completion_a",
        completion_b_col: str = "completion_b",
        keep_features: Optional[Path] = None,
    ):
        self.columns = list(df.columns)
        self.prompts: list[str] = df[prompt_col].to_list()
        self.completion_a: list[str] = df[completion_a_col].to_list()
        self.completion_b: list[str] = df[completion_b_col].to_list()
        logging.info(f"Found {len(self.prompts)} prompts with cols: {self.columns}")

        self.keep_features: Optional[Path] = keep_features
        if self.keep_features:
            self.keep_features.mkdir(parents=True, exist_ok=True)
            logging.info(f"Will save all collected features to {self.keep_features}")

        # Register all extractors here with a shorthand name
        self.REGISTERED_EXTRACTORS = {
            "identity": self._extract_identity,
        }

    def __call__(
        self, features: list[str], threshold: float = 1.0, skip_if_error: bool = True
    ) -> "pd.DataFrame":
        """Extract features from a dataframe

        features (list[str]): list of features to extract.
        threshold (float): number of active features for an instance to swap with human preferences.
        skip_if_error (bool): if set, will skip if an extractor encounters an error.
        RETURN (pd.DataFrame): a dataframe with additional columns 'pref', 'is_swapped', and 'features_used'
        """
        # boolean matrix of size (n_instances, n_feats)
        result_matrix: list[list[int]] = []
        n_features = 0
        for feature in features:
            key, params = self.parse_feature(feature)
            if key in self.REGISTERED_EXTRACTORS:
                logging.info(f"Extracting '{key}' with params: {params}")
                fn = self.REGISTERED_EXTRACTORS[key]

                try:
                    results = fn(**params)
                    result_matrix.append(results)
                except Exception as e:
                    logging.error(f"Error encountered for {key} ({params}): {e}")
                    if skip_if_error:
                        # Skip to the next iteration if an error occurs
                        logging.info("Skipping this feature because skip_if_error=True")
                        continue
                    else:
                        raise
                else:
                    n_features += 1

        # Get all instances that fulfills all (or some) values
        n_active_to_pass = math.floor(n_features * threshold)
        logging.info(
            f"Getting instances. Needs {n_active_to_pass}/{n_features} to swap to human preferences."
        )
        breakpoint()

        # TODO: swap features (take note of the features too)

    def parse_feature(self, s: str) -> tuple[str, dict[str, Any]]:
        if "::" in s:
            key, params_str = s.split("::")
            params = dict(item.split("=") for item in params_str.split(","))
            params = {k: int(v) if v.isdigit() else v for k, v in params.items()}
        else:
            key, params = s, {}
        return key, params

    def _extract_identity(self, **kwargs) -> list[int]:
        return [1 for _ in range(len(self.prompts))]
