"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""

import os

from .neumf_dataset import NeuMFDataset
from ..data_manager import IncrementalSparseMatrix
from ..data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from ..data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from ..base.recommender_utils import reshapeSparse


class Movielens1MReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, dataset_path, pre_splitted_path):

        super(Movielens1MReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Dataset_Movielens1M: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            print("Dataset_Movielens1M: Pre-splitted data not found, building new one")

            # Ensure file is loaded as matrix
            NeuMFDataset.load_rating_file_as_list = NeuMFDataset.load_rating_file_as_matrix

            dataset = NeuMFDataset(dataset_path + "ml-1m")

            URM_train_original, URM_test = dataset.trainMatrix, dataset.testRatings

            URM_train_original = URM_train_original.tocsr()
            URM_test = URM_test.tocsr()

            shape = (max(URM_train_original.shape[0], URM_test.shape[0]),
                     max(URM_train_original.shape[1], URM_test.shape[1]))

            URM_train_original = reshapeSparse(URM_train_original, shape)
            URM_test = reshapeSparse(URM_test, shape)

            URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=shape[0], n_cols=shape[1])

            for user_index in range(len(dataset.testNegatives)):

                user_test_items = dataset.testNegatives[user_index]

                URM_test_negatives_builder.add_single_row(user_index, user_test_items, data=1.0)

            URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()

            URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train_original.copy())

            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

        print("Dataset_Movielens1M: Dataset loaded")
