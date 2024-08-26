import inspect
import itertools
import logging
import math
import random
import re
import sys
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

from src.utils import get_meta_analyzer_features

tqdm_file = sys.stdout
tqdm_bar_format = "{l_bar}{bar}{r_bar}\n"


def get_all_feature_combinations(
    max_number: Optional[int] = None,
    meta_analyzer_n_samples: Optional[int] = None,
) -> tuple[list[str], list[list[str]]]:
    """Get all available feature combinations

    If you include meta_analyzer_n_samples, it will randomly sample different
    combinations of meta-analyzer tags and attach them randomly to some of the
    initial lexical features.

    max_number (Optional[int]): max number of feature combinations
    meta_analyzer_n_samples (Optional[int]): number of meta analyzer features to include for each feature combination
    """
    all_features = [
        mem.removeprefix("_extract_")
        for mem, _ in inspect.getmembers(FeatureExtractor)
        if mem.startswith("_extract")
    ]

    features = [feature for feature in all_features if "analyzer" not in feature]
    if max_number is None:
        max_number = len(features) + 1

    feature_combinations = [
        list(comb)
        for r in range(1, min(max_number + 1, len(features) + 1))
        for comb in tqdm(itertools.combinations(features, r))
    ]

    if meta_analyzer_n_samples:
        logging.info("Adding meta analyzer features")
        # Add meta analyzer features
        meta_analyzer_features = []
        for _ in range(meta_analyzer_n_samples):
            meta_analyzer_features.append(
                list(random.choice(v) for v in get_meta_analyzer_features().values())
            )

        # Sample a few more, sometimes, we don't need the complete feature set
        meta_analyzer_features = [
            random.sample(inner, random.randint(1, len(inner)))
            for inner in meta_analyzer_features
        ]

        # Let's split the meta_analyzer_features. The first half we can append to the lexical
        # features, and the last half we can append as-is.
        split_index = int(0.5 * len(meta_analyzer_features))
        meta_analyzer_features_init_50 = meta_analyzer_features[:split_index]
        meta_analyzer_features_last_50 = meta_analyzer_features[split_index:]

        # For each list in feature_combinations, append a random number of elements
        # (between 1 and the length of the corresponding list) from meta_analyzer_features
        for i in range(
            min(len(feature_combinations), len(meta_analyzer_features_init_50))
        ):
            feature_combinations[i].extend(
                random.sample(
                    meta_analyzer_features_init_50[i],
                    random.randint(1, len(meta_analyzer_features_init_50[i])),
                )
            )

        # Let's also add some features that's just purely from the meta_analyzer
        feature_combinations += meta_analyzer_features_last_50

    return all_features, feature_combinations


def check_lists(query: list[Any], constraint: list[Any], strict: bool = False) -> int:
    """Check if at least one value in query is in constraint.
    If strict=True, then ensures all values of query are in constraint.
    """
    if not query or not constraint:
        return 0

    # Normalize strings
    query = [q.lower() for q in query]
    constraint = [c.lower() for c in constraint]

    if strict:
        return int(all(elem in constraint for elem in query))
    return int(any(elem in constraint for elem in query))


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
        use_cache: bool = True,
    ):
        self._df = df
        self.columns = list(df.columns)
        self.id: list[str] = df[id_col].to_list()
        self.prompts: list[str] = df[prompt_col].to_list()
        self.completions_a: list[str] = df[completion_a_col].to_list()
        self.completions_b: list[str] = df[completion_b_col].to_list()
        # Preferences
        self.pref_humans = df["pref_human"].to_list()
        self.pref_gpt4 = df["pref_gpt4"].to_list()
        logging.info(f"Found {len(self.prompts)} prompts with cols: {self.columns}")

        self.keep_features: Optional[Path] = keep_features
        if self.keep_features:
            self.keep_features.mkdir(parents=True, exist_ok=True)
            logging.info(f"Will save all collected features to {self.keep_features}")

        # Register all extractors here with a shorthand name
        self.REGISTERED_EXTRACTORS = {
            "random": self._extract_random,
            "entity_sim": self._extract_entity_sim,
            "bertscore": self._extract_bertscore,
            "bertscore_length": self._extract_bertscore_length,
            "cosine_sim": self._extract_cosine_sim,
            "rouge": self._extract_rouge,
            "analyzer_closed_set": self._extract_analyzer_closed_set,
            "analyzer_scalar": self._extract_analyzer_scalar,
            "analyzer_open_set": self._extract_analyzer_open_set,
        }

        # Cache data structure
        self.use_cache = use_cache
        self.cache: dict[str, Any] = {}

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
                if params:
                    logging.info(f"Extracting '{key}' with params: {params}")
                else:
                    logging.info(f"Extracting '{key}' with default params")
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
            f"Getting instances. Needs at least {n_active_to_pass}/{n_features} to swap to human preferences."
        )
        result_matrix = np.array(result_matrix)
        if len(result_matrix) == 0:
            # Handle error
            logging.info(
                "Didn't find any features for this combination. Will return GPT-4 preferences"
            )
            to_swap = [False for _ in range(len(self.pref_gpt4))]
        else:
            n_active_features = np.sum(result_matrix, axis=0)
            to_swap = n_active_features >= n_active_to_pass
            logging.info(f"Swapping {sum(to_swap)} samples with human preferences.")

        prefs = [
            human if swap else gpt4
            for human, gpt4, swap in zip(self.pref_humans, self.pref_gpt4, to_swap)
        ]
        df = pd.DataFrame(
            {
                "id": self.id,
                "prompt": self.prompts,
                "completion_a": self.completions_a,
                "completion_b": self.completions_b,
                "is_swapped": list(to_swap),
                "features_used": ",".join(features),
                "pref": prefs,
            }
        )
        return df

    def _save_features(self, output_path: Path, extra_columns: dict[str, Any]):
        dataset = {
            "id": self.id,
            "prompt": self.prompts,
            "completion_a": self.completions_a,
            "completion_b": self.completions_b,
        }
        dataset.update(extra_columns)
        output_df = pd.DataFrame(dataset)
        output_df.to_json(output_path, lines=True, orient="records")

    def _cache_result(self, key: str, scores: list[Any]):
        self.cache[key] = scores

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
            params = dict(item.split("=") for item in params_str.split("|"))
            params = {k: _convert(v) for k, v in params.items()}
        else:
            key, params = s, {}
        return key, params

    def _extract_random(self, threshold: float = 0.5, **kwargs) -> list[bool]:
        return [1 if random.random() >= 0.5 else 0 for _ in range(len(self.prompts))]

    def _extract_entity_sim(
        self,
        threshold: float = 0.8,
        model_name: str = "en_core_web_lg",
        n_process: int = 4,
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = "entity_sim"

        if FEATURE_NAME in self.cache and self.use_cache:
            logging.info(f"Using cached results for {FEATURE_NAME}")
            scores = self.cache[FEATURE_NAME]
        else:
            model = spacy.load(model_name)
            lemmatizer = WordNetLemmatizer()

            docs_a = model.pipe(self.completions_a, n_process=n_process)
            docs_b = model.pipe(self.completions_b, n_process=n_process)
            scores = []

            for doc_a, doc_b in tqdm(
                zip(docs_a, docs_b),
                file=tqdm_file,
                bar_format=tqdm_bar_format,
                total=len(self.completions_a),
            ):
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
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={FEATURE_NAME: scores},
            )

        if self.use_cache:
            self._cache_result(key=FEATURE_NAME, scores=scores)

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_bertscore(
        self,
        model_type: str = "distilbert-base-uncased",
        threshold: float = 0.8,
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = "bertscore"

        if FEATURE_NAME in self.cache and self.use_cache:
            logging.info(f"Using cached results for {FEATURE_NAME}")
            scores = self.cache[FEATURE_NAME]
        else:
            bertscore = evaluate.load("bertscore")
            scores = bertscore.compute(
                predictions=self.completions_a,
                references=self.completions_b,
                verbose=True,
                use_fast_tokenizer=True,
                nthreads=8,
                device="cuda",
                model_type=model_type,
            )["f1"]

        if self.keep_features:
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={FEATURE_NAME: scores},
            )

        if self.use_cache:
            self._cache_result(key=FEATURE_NAME, scores=scores)

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_bertscore_length(
        self,
        threshold: float = 0.40,
        model_type: str = "distilbert-base-uncased",
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = "bertscore_length"

        if FEATURE_NAME in self.cache and self.use_cache:
            logging.info(f"Using cached results for {FEATURE_NAME}")
            scores = self.cache[FEATURE_NAME]
        else:
            length_penalties = []
            if "bertscore" in self.cache:
                logging.info("Using cached bertscore results")
                bert_scores = self.cache["bertscore"]
            else:
                # Compute the result
                bertscore = evaluate.load("bertscore")
                bert_scores = bertscore.compute(
                    predictions=self.completions_a,
                    references=self.completions_b,
                    lang="en",
                    verbose=True,
                    use_fast_tokenizer=True,
                    nthreads=8,
                    device="cuda",
                    model_type=model_type,
                )["f1"]

                if self.use_cache:
                    self._cache_result(key="bertscore", scores=bert_scores)

            logging.info("Computing length penalties")
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
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={FEATURE_NAME: scores},
            )

        if self.use_cache:
            self._cache_result(key=FEATURE_NAME, scores=scores)

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_rouge(self, threshold: float = 0.4, **kwargs) -> list[bool]:
        FEATURE_NAME = "rouge"

        if FEATURE_NAME in self.cache and self.use_cache:
            logging.info(f"Using cached results for {FEATURE_NAME}")
            scores = self.cache[FEATURE_NAME]
        else:
            rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
            scores = []
            for a, b in tqdm(
                zip(self.completions_a, self.completions_b),
                file=tqdm_file,
                bar_format=tqdm_bar_format,
                total=len(self.completions_a),
            ):
                score = rouge.score(prediction=a, target=b)["rouge1"].fmeasure
                scores.append(score)

        if self.keep_features:
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={FEATURE_NAME: scores},
            )

        if self.use_cache:
            self._cache_result(key=FEATURE_NAME, scores=scores)

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

        if FEATURE_NAME in self.cache and self.use_cache:
            logging.info(f"Using cached results for {FEATURE_NAME}")
            scores = self.cache[FEATURE_NAME]
        else:
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
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={FEATURE_NAME: scores},
            )

        if self.use_cache:
            self._cache_result(key=FEATURE_NAME, scores=scores)

        logging.info(f"Filtering instances where score > {threshold}")
        return [1 if score >= threshold else 0 for score in scores]

    def _extract_analyzer_closed_set(
        self,
        feature_name: str = "subject_of_expertise",
        constraints: str = "Computer sciences,Mathematics",
        strict: bool = False,
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = feature_name
        if feature_name not in self.columns:
            raise ValueError(
                f"No `{feature_name}` field found in the dataset! Skipping this feature"
            )

        # No caching here
        include_list = [domain.strip() for domain in constraints.split(",")]
        instance_features = self._df[feature_name].to_list()
        scores = [
            check_lists(query=feat, constraint=include_list, strict=strict)
            for feat in instance_features
        ]

        if self.keep_features:
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={
                    FEATURE_NAME: scores,
                    f"{FEATURE_NAME}_closed_set": constraints,
                },
            )

        return scores

    def _extract_analyzer_scalar(
        self,
        feature_name: str = "expertise_level",
        value: str = "general public",
        **kwargs,
    ) -> list[bool]:
        FEATURE_NAME = feature_name
        if feature_name not in self.columns:
            raise ValueError(
                f"No `{feature_name}` field found in the dataset! Skipping this feature"
            )

        # No caching here
        instance_features = self._df[feature_name].to_list()
        scores = [feat.lower() == value.lower() for feat in instance_features]

        if self.keep_features:
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={
                    FEATURE_NAME: scores,
                    f"{FEATURE_NAME}_scalar": value,
                },
            )

        return scores

    def _extract_analyzer_open_set(
        self,
        feature_name: str = "format_constraints",
        check_for_existence: bool = True,
        constraints: Optional[str] = None,
        strict: bool = False,
        **kwargs,
    ):
        FEATURE_NAME = feature_name
        if feature_name not in self.columns:
            raise ValueError(
                f"No `{feature_name}` field found in the dataset! Skipping this feature"
            )

        # No caching here
        instance_features = self._df[feature_name].to_list()
        if not check_for_existence and not constraints:
            raise ValueError(
                "Must pass a value to `value` or set `check_for_instance` to True"
            )

        if check_for_existence:
            scores = [1 if feat else 0 for feat in instance_features]
        if constraints:
            include_list = [constraint.strip() for constraint in constraints.split(",")]
            scores = [
                check_lists(query=feat, constraint=include_list, strict=strict)
                for feat in instance_features
            ]

        if self.keep_features:
            self._save_features(
                output_path=self.keep_features / f"{FEATURE_NAME}.jsonl",
                extra_columns={
                    FEATURE_NAME: scores,
                    f"{FEATURE_NAME}_scalar": constraints,
                },
            )

        return scores
