"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""
from typing import Optional, List, Union, Tuple

import numpy as np

from .data_io import DataIO
from .recommender_utils import check_matrix


class BaseRecommender(object):
    """Abstract BaseRecommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self, URM_train, verbose=True):
        """
        Parameters
        ----------
        URM_train : [TODO:type]
            User-Item interaction matrix
        verbose : bool, optional
            control verbosity, by default True
        """
        super().__init__()

        # Convert matrix to csr and eliminate zeros
        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        # Detect empty (user) rows
        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(),
                self._cold_user_mask.sum() / self.n_users * 100))

        # Detect empty (item) columns
        # This could be an expensive operation?
        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold items.".format(
                self._cold_item_mask.sum(),
                self._cold_item_mask.sum() / self.n_items * 100
            ))

    def _get_cold_user_mask(self):
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        return self._cold_item_mask

    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_URM_train(self, URM_train_new, **kwargs):
        # TODO: refactor the use this also in __init__
        assert self.URM_train.shape == URM_train_new.shape, \
            "{}: set_URM_train old and new URM train have different shapes".format(
                self.RECOMMENDER_NAME)

        if len(kwargs) > 0:
            self._print(
                "set_URM_train keyword arguments not supported for this "
                "recommender class. Received: {}".format(kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum() / len(self._cold_user_mask) * 100))

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                     COMPUTE AND FILTER RECOMMENDATION LIST                          ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, user_id, scores):
        """Set seen items of `user_id` in `scores` to -np.inf
        """
        assert self.URM_train.getformat() == "csr", \
            "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"
        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """Get item scores for the (user_id, item_id) pairs

        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        raise NotImplementedError(
            "BaseRecommender: compute_item_score not assigned for current recommender, "
            "unable to compute prediction scores"
        )

    def recommend(
        self, user_id_array: np.array, cutoff: Optional[int] = None,
        remove_seen_flag: bool = True, items_to_compute: Optional[np.array] = None,
        remove_top_pop_flag: bool = False, remove_custom_items_flag: bool = False,
        return_scores: bool = False
    ) -> Union[Union[List[List[int]], List[int]], Union[List[List[int]], List[int]], np.array]:
        """Make recommendations for users in user_id_array

        Parameters
        ----------
        user_id_array : np.array
            List of user_ids to make recommendations for
        cutoff : Optional[int], optional
            [description], by default None
        remove_seen_flag : bool, optional
            Don't recommend seen items, by default True
        items_to_compute : np.array, optional
            List of item_ids to include in computation, by default None
        remove_top_pop_flag : bool, optional
            Don't recommend top popular items, by default False
        remove_custom_items_flag : bool, optional
            [description], by default False
        return_scores : bool, optional
            Also return the score matrix, by default False

        Returns
        -------
        Union[List, Tuple[List, np.array]]
            ranking_list or (ranking_list, scores_batch) accorgin to return_scores
        """
        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            # No cutoff, take all items
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]

            if remove_seen_flag:
                # Don't recommend seen items
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index

        # relevant_items_partition is block_size x cutoff
        # (np.partition: kth item in the results will be where it would be in a sorted array
        #  and the smaller elements to the left and equal or greater elements to the right)
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0: cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        # The ranking matrix holds Top-K item ids for each user id
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None],
            relevant_items_partition_sorting
        ]

        ranking_list: List[List[int]] = [[]] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

        # Return single list for one user, instead of list of lists
        ranking_list_final: Union[List[List[int]], List[int]] = ranking_list
        if single_user:
            ranking_list_final = ranking_list[0]

        if return_scores:
            return ranking_list_final, scores_batch

        else:
            return ranking_list_final

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def save_model(self, folder_path, file_name=None):
        raise NotImplementedError("BaseRecommender: save_model not implemented")

    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        # Set attributes of the instance to values in `data_dict`
        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        self._print("Loading complete")
