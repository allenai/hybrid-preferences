import sys
import argparse
from beaker import Beaker, Experiment
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Get results from Beaker that evaluates on RewardBench"
    parser = argparse.ArgumentParser()
    parser.add_argument("--beaker_workspace", default="ai2/ljm-oe-adapt", help="Beaker workspace to fetch experiments")
    parser.add_argument("--experiment_prefix", default="rm-eval-", help="Prefix for experiments to fetch")
    # fmt:on
    return parser.parse_args()


def main():
    args = get_args()
    beaker = Beaker.from_env(default_workspace=args.beaker_workspace)
    try:
        account = beaker.account.whoami()
    except Exception as e:
        logging.error(f"Please authenticate using `beaker account login`: {e}")
        raise
    else:
        logging.info(f"Logged-in as {account.name} ({account.email})")

    experiments = [
        experiment
        for experiment in beaker.workspace.experiments(
            args.beaker_workspace,
            match=args.experiment_prefix,
        )
        if is_done(experiment)
    ]
    logging.info(
        f"Found {len(experiments)} experiments that match '{args.experiment_prefix}'"
    )
    breakpoint()


def is_done(experiment: "Experiment") -> bool:
    return True if experiment.jobs[0].status.finalized else False


if __name__ == "__main__":
    main()
