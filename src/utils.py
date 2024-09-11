import random


def find_meta_category(query: str) -> str:
    for category, inner_dict in meta_analyzer_values.items():
        if query in inner_dict:
            return category
    return None


def tag_to_normal(feat: str) -> str:
    """Convert tags used for data model to tags used for results"""
    if "analyzer" in feat:
        feature_name_str, value_str = feat.split("::")[1].split("|")
        feature_name, value = (
            feature_name_str.split("=")[-1],
            value_str.split("=")[-1],
        )
        return f"{feature_name}={value}"
    else:
        return feat.replace("::", "__")


def get_meta_analyzer_features(**kwargs) -> dict[str, list[str]]:
    """Get meta-analyzer features

    If you want to sample a specific feature, simply pass a keyword argument with the
    key as the feature name (e.g., subject_of_expertise) and value as an int (or the number to sample).
    This only works for closed_set types of features.
    """
    feature_params = {}

    # Get all possible feature params first
    for extractor_name, features in meta_analyzer_values.items():
        if extractor_name == "closed_set":
            # Add features for closed_set type tags
            # For now, we set strict=False. We basically just want that the tags
            # fall under on of the values we set.
            for feature, values in features.items():
                for value in values:
                    func_name = f"analyzer_{extractor_name}::feature_name={feature}|constraints={value}"
                    if feature not in feature_params:
                        feature_params[feature] = [func_name]
                    else:
                        feature_params[feature].append(func_name)

                # Sampling for closed_sets
                if feature in kwargs:
                    n_samples = kwargs[feature]
                    all_features = feature_params[feature]
                    print(f"Sampling {n_samples} features for closed_set '{feature}'")
                    feature_params[feature] = random.sample(
                        all_features, min(n_samples, len(all_features))
                    )
        elif extractor_name == "scalar":
            # For scalar features, we check each value
            for feature, values in features.items():
                for value in values:
                    func_name = f"analyzer_{extractor_name}::feature_name={feature}|value={value}"
                    if feature not in feature_params:
                        feature_params[feature] = [func_name]
                    else:
                        feature_params[feature].append(func_name)
        elif extractor_name == "open_set":
            # For open_set values, we just check for existence
            for feature, values in features.items():
                func_name = f"analyzer_{extractor_name}::feature_name={feature}|check_for_existence=1"
                if feature not in feature_params:
                    feature_params[feature] = [func_name]
                else:
                    feature_params[feature].append(func_name)
        else:
            raise ValueError(f"Unknown extractor name: {extractor_name}")

    return feature_params


def find_meta_category(query: str) -> str:
    for category, inner_dict in meta_analyzer_values.items():
        if query in inner_dict:
            return category
    return None


meta_analyzer_values = {
    "closed_set": {
        # Reduce this number
        "subject_of_expertise": [
            "Anthropology",
            "History",
            "Linguistics and language",
            "Philosophy",
            "Religion",
            "Economics",
            "Geography",
            "Political science",
            "Psychology",
            "Sociology",
            "Biology",
            "Chemistry",
            "Earth sciences",
            "Physics",
            "Space sciences",
            "Computer sciences",
            "Logic",
            "System science",
            "Agriculture",
            "Architecture and design",
            "Business",
            "Divinity",
            "Education",
            "Chemical engineering",
            "Civil engineering",
            "Electrical engineering",
            "Materials science and engineering",
            "Mechanical engineering",
            "Environmental studies and forestry",
            "Family and consumer science",
            "Human physical performance and recreation",
            "Journalism",
            "Media studies and communication",
            "Law",
            "Library and museum studies",
            "Medicine and health",
            "Military sciences",
            "Social work",
            "Transportation",
            "Culinary arts",
            "Literature",
            "Performing arts",
            "Visual arts",
            "Mathematics",
            "Public administration",
            "Others",
        ],
        "languages": ["English"],
    },
    "scalar": {
        "expertise_level": [
            "general public",
            "basic domain knowledge",
            "expert domain knowledge",
        ],
        "open_endedness": [
            "no",
            "low",
            "moderate",
            "high",
        ],
        "safety_concern": [
            "safe",
            "low",
            "moderate",
            "high",
        ],
        "complexity_of_intents": [
            "simple",
            "moderate",
            "complex",
        ],
    },
    "open_set": {
        "type_of_in_context_material": [
            # List of values
            "url",
            "table",
            "options",
        ],
        "format_constraints": [
            # List of values
            "code",
            "table",
        ],
    },
}
