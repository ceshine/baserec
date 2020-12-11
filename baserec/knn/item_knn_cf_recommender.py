"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""
import numpy as np

from baserec.base.recommender_utils import check_matrix
from baserec.base.base_similarity_matrix_recommender import BaseItemSimilarityMatrixRecommender
from baserec.base.ir_feature_weighting import okapi_BM_25, TF_IDF
from baserec.base.similarity.compute_similarity import ComputeSimilarity


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender (based on item-item similarities)"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, verbose=True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        # URM_train shape: (U x I)
        # flipped : (I x U) -> re-weight along I (discount users that rates too many items)
        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = ComputeSimilarity(
            self.URM_train, shrink=shrink, topK=topK,
            normalize=normalize, similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
