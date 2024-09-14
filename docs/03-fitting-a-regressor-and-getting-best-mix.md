# Fitting a regressor and getting the best mix

> In this guide, we will fit a regressor using the **proxy reward model performance**, and then use the fitted model to get the best mix by computing the expected gain for each instance.
> In the end, we'll get the preference mix for any given budget that we set.

## Fitting a regression model

From the previous guide, we [generated a CSV file containing the features and the overall RewardBench performance](https://github.com/allenai/human-pref-datamodel/blob/main/docs/02-evaluating-proxy-reward-models.md#fetch-all-results-and-combine-it-with-the-features) using `scripts/fetch_evals_rewardbench.py`.
This time, we'll use that file to train a linear regressor:

```sh
python3 -m scripts.train_regressor \
    --input_path path/to/eval-runs.py \
    --output_dir path/to/output/ \
    --model linear \
    --random_seed 42
```

This script will output the model, `model.pkl` and the feature coefficients or in this case the weights of the linear model, `coef.jsonl`.

## Sampling subsets

Now that we have a model, we can compute the gain of using the human annotation for each instance.

Let me talk a bit about how we get the best subset:

- Each instance will have a gain, and then we sort them in descending order. We compute the gain by first converting the lexical and metadata features into binary (one-hot encoding), and multiplying it with the coefficients from `model.pkl`.
- For a given `budget`, we take the top-`budget` instances with the highest gain and add swap human annotations for them.
- Then we train reward models and measure their actual performance.

We do all this using the command below:

```sh
python3 -m scripts.sample_best_subset \
    --input_path path/to/features.jsonl \
    --output_dir path/to/output/directory/ \
    --model_path path/to/model.pkl \
    --budget 0.25 0.50 0.75
```

Here, a budget of `0.25` means "we want 25% of the whole dataset to be annotated by humans."
You can also pass a whole number, but it's much easier to think in terms of proportions.

The output directory in `--output_dir` will contain the features and actual swapped subsets as we had when we're generating [proxy reward models in Step 1](https://github.com/allenai/human-pref-datamodel/blob/main/docs/01-training-proxy-reward-models.md#create-proxy-dpo-training-datasets).
Similar to that step, you can should upload the swapped subsets to Google Cloud Storage and [run them in TPUs](https://github.com/allenai/human-pref-datamodel/blob/main/docs/01-training-proxy-reward-models.md#train-reward-models-on-a-tpu).
