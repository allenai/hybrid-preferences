import logging
from typing import Optional

import pandas as pd


class FeatureExtractor:

    REGISTERED_EXTRACTORS = {
        "identity": _extract_identity,
    }

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

    def __call__(self, features: list[str]) -> "pd.DataFrame":
        pass

    def _extract_identity(self):
        pass
