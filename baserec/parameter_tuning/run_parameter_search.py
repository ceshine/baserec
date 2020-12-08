"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""

import os
import traceback
import multiprocessing
from functools import partial

from skopt.space import Real, Integer, Categorical

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from baserec.base.non_personalized_recommenders import TopPop, Random, GlobalEffects

from baserec.ease_r import EASE_R_Recommender
from baserec.matrix_factorization import IALSRecommender
from baserec.slim_bpr import SlimBprCython

######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
# from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
# from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender


######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
# from KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
# from KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender


######################################################################


from .search_bayesian_skopt import SearchBayesianSkopt
from .search_single_case import SearchSingleCase
from .search_abstract_class import SearchInputRecommenderArgs


def run_search_collaborative(recommender_class, URM_train, URM_train_last_test=None,
                             n_cases=35, n_random_starts=5, resume_from_saved=False,
                             save_model="best", evaluate_on_test="best",
                             evaluator_validation=None, evaluator_test=None, evaluator_validation_earlystopping=None,
                             metric_to_optimize="PRECISION",
                             output_folder_path="result_experiments/", parallelizeKNN=True,
                             allow_weighting=True, similarity_type_list=None):
    """
    This function performs the hyperparameter optimization for a collaborative recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see ParameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see ParameterTuning/SearchAbstractClass for details
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "evaluator_object": evaluator_validation_earlystopping,
        "lower_validations_allowed": 5,
        "validation_metric": metric_to_optimize,
    }

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameter_searcher = SearchBayesianSkopt(
            recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no parameters therefore only one evaluation is needed
            """

            parameter_searcher = SearchSingleCase(
                recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )

            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None

            parameter_searcher.search(recommender_input_args,
                                      recommender_input_args_last_test=recommender_input_args_last_test,
                                      fit_hyperparameters_values={},
                                      output_folder_path=output_folder_path,
                                      output_file_name_root=output_file_name_root,
                                      resume_from_saved=resume_from_saved,
                                      save_model=save_model,
                                      evaluate_on_test=evaluate_on_test,
                                      )

            return

        ##########################################################################################################

        # if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

        #     if similarity_type_list is None:
        #         similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        #     if URM_train_last_test is not None:
        #         recommender_input_args_last_test = recommender_input_args.copy()
        #         recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        #     else:
        #         recommender_input_args_last_test = None

        #     run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
        #                                                               recommender_input_args=recommender_input_args,
        #                                                               parameter_search_space={},
        #                                                               parameterSearch=parameterSearch,
        #                                                               n_cases=n_cases,
        #                                                               n_random_starts=n_random_starts,
        #                                                               resume_from_saved=resume_from_saved,
        #                                                               save_model=save_model,
        #                                                               evaluate_on_test=evaluate_on_test,
        #                                                               output_folder_path=output_folder_path,
        #                                                               output_file_name_root=output_file_name_root,
        #                                                               metric_to_optimize=metric_to_optimize,
        #                                                               allow_weighting=allow_weighting,
        #                                                               recommender_input_args_last_test=recommender_input_args_last_test)

            # if parallelizeKNN:
            #     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            #     pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            #     pool.close()
            #     pool.join()

            # else:

            #     for similarity_type in similarity_type_list:
            #         run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

            # return

        ##########################################################################################################

        # if recommender_class is P3alphaRecommender:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
        #     hyperparameters_range_dictionary["alpha"] = Real(low=0, high=2, prior='uniform')
        #     hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        ##########################################################################################################

        # if recommender_class is RP3betaRecommender:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
        #     hyperparameters_range_dictionary["alpha"] = Real(low=0, high=2, prior='uniform')
        #     hyperparameters_range_dictionary["beta"] = Real(low=0, high=2, prior='uniform')
        #     hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        ##########################################################################################################

        # if recommender_class is MatrixFactorization_FunkSVD_Cython:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
        #     hyperparameters_range_dictionary["epochs"] = Categorical([500])
        #     hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
        #     hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        #     hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
        #     hyperparameters_range_dictionary["item_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["user_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-4, high=1e-1, prior='log-uniform')
        #     hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low=0.0, high=0.5, prior='uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS=earlystopping_keywargs
        #     )

        ##########################################################################################################

        # if recommender_class is MatrixFactorization_AsySVD_Cython:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
        #     hyperparameters_range_dictionary["epochs"] = Categorical([500])
        #     hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
        #     hyperparameters_range_dictionary["batch_size"] = Categorical([1])
        #     hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
        #     hyperparameters_range_dictionary["item_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["user_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-4, high=1e-1, prior='log-uniform')
        #     hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low=0.0, high=0.5, prior='uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS=earlystopping_keywargs
        #     )

        ##########################################################################################################

        # if recommender_class is MatrixFactorization_BPR_Cython:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
        #     hyperparameters_range_dictionary["epochs"] = Categorical([1500])
        #     hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
        #     hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        #     hyperparameters_range_dictionary["positive_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["negative_reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
        #     hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-4, high=1e-1, prior='log-uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={**earlystopping_keywargs,
        #                           "positive_threshold_BPR": None}
        #     )

        ##########################################################################################################

        if recommender_class is IALSRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["confidence_scaling"] = Categorical(["linear", "log"])
            hyperparameters_range_dictionary["alpha"] = Real(low=1e-3, high=50.0, prior='log-uniform')
            hyperparameters_range_dictionary["epsilon"] = Real(low=1e-3, high=10.0, prior='log-uniform')
            hyperparameters_range_dictionary["reg"] = Real(low=1e-5, high=1e-2, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                # FIT_KEYWORD_ARGS=earlystopping_keywargs
            )

        ##########################################################################################################

        # if recommender_class is PureSVDRecommender:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        ##########################################################################################################

        # if recommender_class is NMFRecommender:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)
        #     hyperparameters_range_dictionary["solver"] = Categorical(["coordinate_descent", "multiplicative_update"])
        #     hyperparameters_range_dictionary["init_type"] = Categorical(["random", "nndsvda"])
        #     hyperparameters_range_dictionary["beta_loss"] = Categorical(["frobenius", "kullback-leibler"])

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        #########################################################################################################

        if recommender_class is SlimBprCython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["epochs"] = Categorical([1500])
            hyperparameters_range_dictionary["symmetric"] = Categorical([True, False])
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["lambda_i"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
            hyperparameters_range_dictionary["lambda_j"] = Real(low=1e-5, high=1e-2, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-4, high=1e-1, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={**earlystopping_keywargs,
                                  "positive_threshold_BPR": None,
                                  'train_with_sparse_weights': None}
            )

        ##########################################################################################################

        # if recommender_class is SLIMElasticNetRecommender:

        #     hyperparameters_range_dictionary = {}
        #     hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
        #     hyperparameters_range_dictionary["l1_ratio"] = Real(low=1e-5, high=1.0, prior='log-uniform')
        #     hyperparameters_range_dictionary["alpha"] = Real(low=1e-3, high=1.0, prior='uniform')

        #     recommender_input_args = SearchInputRecommenderArgs(
        #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        #         CONSTRUCTOR_KEYWORD_ARGS={},
        #         FIT_POSITIONAL_ARGS=[],
        #         FIT_KEYWORD_ARGS={}
        #     )

        #########################################################################################################

        if recommender_class is EASE_R_Recommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Categorical([None])  # Integer(5, 3000)
            hyperparameters_range_dictionary["normalize_matrix"] = Categorical([False])
            hyperparameters_range_dictionary["l2_norm"] = Real(low=1e0, high=1e7, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={}
            )

        #########################################################################################################

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        # Final step, after the hyperparameter range has been defined for each type of algorithm
        parameter_searcher.search(recommender_input_args,
                                  parameter_search_space=hyperparameters_range_dictionary,
                                  n_cases=n_cases,
                                  n_random_starts=n_random_starts,
                                  resume_from_saved=resume_from_saved,
                                  save_model=save_model,
                                  evaluate_on_test=evaluate_on_test,
                                  output_folder_path=output_folder_path,
                                  output_file_name_root=output_file_name_root,
                                  metric_to_optimize=metric_to_optimize,
                                  recommender_input_args_last_test=recommender_input_args_last_test)

    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()
