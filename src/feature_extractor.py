import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import evaluate
import numpy as np
import pandas as pd
import spacy
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


class FeatureExtractor:
    """Feature extractor class that takes in a dataframe of prompts, completions, and other metadata

    Each extractor returns a boolean list, indicating if a certain instance fulfills a requirement.
    You can set a `threshold` in the __call__ function to control how many features need to be active.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        *,
        id_col: str = "id",
        prompt_col: str = "text",
        completion_a_col: str = "completion_a",
        completion_b_col: str = "completion_b",
        keep_features: Optional[Path] = None,
    ):
        self.columns = list(df.columns)
        self.id: list[str] = df[id_col].to_list()
        self.prompts: list[str] = df[prompt_col].to_list()
        self.completions_a: list[str] = df[completion_a_col].to_list()
        self.completions_b: list[str] = df[completion_b_col].to_list()
        logging.info(f"Found {len(self.prompts)} prompts with cols: {self.columns}")

        self.keep_features: Optional[Path] = keep_features
        if self.keep_features:
            self.keep_features.mkdir(parents=True, exist_ok=True)
            logging.info(f"Will save all collected features to {self.keep_features}")

        # Register all extractors here with a shorthand name
        self.REGISTERED_EXTRACTORS = {
            "identity": self._extract_identity,
            "entity_sim": self._extract_entity_sim,
            "bertscore": self._extract_bertscore,
            "bertscore_length": self._extract_bertscore_length,
            "cosine_sim": self._extract_cosine_sim,
            "rouge": self._extract_rouge,
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

    def save_features(self, output_path: Path, extra_columns: dict[str, Any]):
        dataset = {
            "id": self.id,
            "prompt": self.prompts,
            "completion_a": self.completions_a,
            "completion_b": self.completions_b,
        }
        dataset.update(extra_columns)
        output_df = pd.DataFrame(dataset)
        output_df.to_json(output_path, lines=True, orient="records")

    def parse_feature(self, s: str) -> tuple[str, dict[str, Any]]:
        def _convert(v):
            if v.isdigit():
                return int(v)
            try:
                return float(v)
            except ValueError:
                return v

        if "::" in s:
            key, params_str = s.split("::")
            params = dict(item.split("=") for item in params_str.split(","))
            params = {k: _convert(v) for k, v in params.items()}
        else:
            key, params = s, {}
        return key, params

    def _extract_identity(self, **kwargs) -> list[bool]:
        return [1 for _ in range(len(self.prompts))]

    def _extract_entity_sim(
        self,
        threshold: float = 0.8,
        model_name: str = "en_core_web_lg",
        n_process: int = 4,
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = "entity_sim"

        model = spacy.load(model_name)
        lemmatizer = WordNetLemmatizer()

        docs_a = model.pipe(self.completions_a, n_process=n_process)
        docs_b = model.pipe(self.completions_b, n_process=n_process)
        scores = []

        for doc_a, doc_b in tqdm(zip(docs_a, docs_b)):
            gen_a_ents = set()
            gen_b_ents = set()

            for ent in doc_a.ents:
                ent_text = re.sub("[^0-9 a-zA-Z]+", "", ent.text)
                ent_text = lemmatizer.lemmatize(ent_text.replace("the", "").strip())
                ent_text = ent_text.lower()

                gen_a_ents.add(ent_text)

            for ent in doc_b.ents:
                ent_text = re.sub("[^0-9 a-zA-Z]+", "", ent.text)
                ent_text = lemmatizer.lemmatize(ent_text.replace("the", "").strip())
                ent_text = ent_text.lower()

                gen_b_ents.add(ent_text)

            intersection = len(gen_a_ents.intersection(gen_b_ents))
            union = (len(gen_b_ents) + len(gen_b_ents)) - intersection

            # If there are no entities in either of the generations, return 1
            score = 1 if union == 0 else intersection / union
            scores.append(score)

        if self.keep_features:
            self.save_features(
                output_path=self.keep_features,
                extra_columns={FEATURE_NAME: scores},
            )

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_bertscore(self, threshold: float = 0.8, **kwargs) -> list[bool]:
        FEATURE_NAME = "bertscore"
        bertscore = evaluate.load("bertscore")
        scores = bertscore.compute(
            predictions=self.completions_a,
            references=self.completions_b,
            lang="en",
            verbose=True,
        )["f1"]

        if self.keep_features:
            self.save_features(
                output_path=self.keep_features,
                extra_columns={FEATURE_NAME: scores},
            )

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_bertscore_length(self, threshold: float = 0.9, **kwargs) -> list[bool]:
        FEATURE_NAME = "bertscore_length"

        length_penalties = []
        bertscore = evaluate.load("bertscore")
        bert_scores = bertscore.compute(
            predictions=self.completions_a,
            references=self.completions_b,
            lang="en",
            verbose=True,
        )["f1"]

        for a, b in zip(self.completions_a, self.completions_b):
            ref, cand = (a, b) if len(a) > len(b) else (b, a)
            try:
                length_penalty = np.exp(
                    1 - len(word_tokenize(ref)) / len(word_tokenize(cand))
                )
            except ZeroDivisionError:
                length_penalty = 0
            length_penalties.append(length_penalty)

        scores = [i * j for i, j in zip(bert_scores, length_penalties)]
        if self.keep_features:
            self.save_features(
                output_path=self.keep_features,
                extra_columns={FEATURE_NAME: scores},
            )

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_rouge(self, threshold: float = 0.8, **kwargs) -> list[bool]:
        FEATURE_NAME = "rouge"

        rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = []
        for a, b in tqdm(zip(self.completions_a, self.completions_b)):
            score = rouge.score(prediction=a, target=b)["rouge1"].fmeasure
            scores.append(score)

        if self.keep_features:
            self.save_features(
                output_path=self.keep_features,
                extra_columns={FEATURE_NAME: scores},
            )

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_cosine_sim(
        self,
        threshold: float = 0.8,
        model_name: str = "all-distilroberta-v1",
        device: str = "cuda",
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = "cosine_sim"

        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = 200

        embeddings_a = model.encode(
            self.completions_a,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device,
        )
        embeddings_b = model.encode(
            self.completions_b,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device,
        )
        cosine_scores = util.cos_sim(embeddings_a, embeddings_b)
        scores = cosine_scores.diag().cpu().numpy().tolist()

        if self.keep_features:
            self.save_features(
                output_path=self.keep_features,
                extra_columns={FEATURE_NAME: scores},
            )

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]
