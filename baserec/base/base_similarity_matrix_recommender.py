"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""
import numpy as np

from .base_recommender import BaseRecommender
from .data_io import DataIO


class BaseSimilarityMatrixRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseSimilarityMatrixRecommender, self).__init__(URM_train, verbose=verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

    def _check_format(self):
        """Display warning if interaction or weight matrix is not in csr format"""
        if not self._URM_train_format_checked:
            if self.URM_train.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, ".format(
                        "URM_train", "csr"
                    ) + "this will significantly slow down the computation."
                )
            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:
            if self.W_sparse.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, ".format(
                        "W_sparse", "csr"
                    ) + "this will significantly slow down the computation.")
            self._W_sparse_format_checked = True

    def save_model(self, folder_path, file_name=None):
        """Only one parameter — W_sparse — to save."""
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_sparse": self.W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


#########################################################################################################
##########                                                                                     ##########
##########                               COMPUTE ITEM SCORES                                   ##########
##########                                                                                     ##########
#########################################################################################################


class BaseItemSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        # doing R @ W (final shape should be U x I)
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores


class BaseUserSimilarityMatrixRecommender(BaseSimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        return item_scores
