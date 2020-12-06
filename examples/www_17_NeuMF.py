"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""
import os
import traceback
from functools import partial

import typer
# import numpy as np

from baserec import TopPop, Random, EASE_R_Recommender
from baserec.dataset_readers import Movielens1MReader, PinterestICCVReader
from baserec.utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from baserec.utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from baserec.base.evaluation.evaluator import EvaluatorNegativeItemSample
from baserec.parameter_tuning.run_parameter_search import run_search_collaborative
CONFERENCE_NAME = "WWW"


def read_data_split_and_search(dataset_name, dataset_path):
    result_folder_path = "result_experiments/{}/{}/".format(CONFERENCE_NAME, dataset_name)

    if dataset_name == "movielens1m":
        dataset = Movielens1MReader(dataset_path, result_folder_path)
    elif dataset_name == "pinterest":
        dataset = PinterestICCVReader(dataset_path, result_folder_path)
    else:
        raise ValueError(f"Unrecognized dataset_name {dataset}")

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data([URM_train, URM_validation, URM_test, URM_test_negative])

    assert_disjoint_matrices([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         result_folder_path + dataset_name + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation + URM_test, URM_train + URM_validation, URM_test],
                               ["Full data", "Training data", "Test data"],
                               result_folder_path + dataset_name + "popularity_statistics")

    collaborative_algorithm_list = [
        Random,
        TopPop,
        # EASE_R_Recommender
    ]

    metric_to_optimize = "HIT_RATE"
    n_cases = 50
    n_random_starts = 15

    evaluator_validation = EvaluatorNegativeItemSample(
        URM_validation, URM_test_negative, cutoff_list=[10])
    evaluator_test = EvaluatorNegativeItemSample(
        URM_test, URM_test_negative, cutoff_list=[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    run_search_collaborative_partial = partial(
        run_search_collaborative,
        URM_train=URM_train,
        URM_train_last_test=URM_train + URM_validation,
        metric_to_optimize=metric_to_optimize,
        evaluator_validation_earlystopping=evaluator_validation,
        evaluator_validation=evaluator_validation,
        evaluator_test=evaluator_test,
        output_folder_path=result_folder_path,
        parallelizeKNN=False,
        allow_weighting=True,
        resume_from_saved=True,
        n_cases=n_cases,
        n_random_starts=n_random_starts)

    for recommender_class in collaborative_algorithm_list:
        try:
            run_search_collaborative_partial(recommender_class)
        except Exception as e:
            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()


def main(dataset: str, dataset_path: str = "data/"):
    assert dataset in ["movielens1m", "pinterest"]

    read_data_split_and_search(dataset, dataset_path)


if __name__ == '__main__':
    typer.run(main)
