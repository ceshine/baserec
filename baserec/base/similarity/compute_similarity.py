"""
@author: Maurizio Ferrari Dacrema & Ceshine Lee
"""
from enum import Enum

import numpy as np
import scipy.sparse as sps

from .compute_similarity_python import ComputeSimilarityPython
from .compute_similarity_euclidean import ComputeSimilarityEuclidean


class SimilarityFunction(Enum):
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"


class ComputeSimilarity:

    def __init__(self, dataMatrix, use_implementation="density", similarity=None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """

        assert np.all(np.isfinite(dataMatrix.data)), \
            "ComputeSimilarity: Data matrix contains {} non finite values".format(
                np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            self.compute_similarity_object = ComputeSimilarityEuclidean(dataMatrix, **args)

        else:

            assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
                "ComputeSimilarity: data has only 1 feature (shape: {}) with dense values," \
                " vector and set based similarities are not defined on 1-dimensional dense data," \
                " use Euclidean similarity instead.".format(dataMatrix.shape)

            if similarity is not None:
                args["similarity"] = similarity

            if use_implementation == "density":

                if isinstance(dataMatrix, np.ndarray):
                    self.dense = True

                elif isinstance(dataMatrix, sps.spmatrix):
                    shape = dataMatrix.shape

                    num_cells = shape[0] * shape[1]

                    sparsity = dataMatrix.nnz / num_cells

                    self.dense = sparsity > 0.5

                else:
                    print("ComputeSimilarity: matrix type not recognized, calling default...")
                    use_implementation = "python"

                if self.dense:
                    print("ComputeSimilarity: detected dense matrix")
                    use_implementation = "python"
                else:
                    use_implementation = "cython"

            if use_implementation == "cython":

                try:
                    from baserec.base.similarity.compute_similarity_cython import ComputeSimilarityCython
                    self.compute_similarity_object = ComputeSimilarityCython(dataMatrix, **args)

                except (ImportError, ModuleNotFoundError):
                    print("Unable to load Cython ComputeSimilarity, reverting to Python")
                    self.compute_similarity_object = ComputeSimilarityPython(dataMatrix, **args)

            elif use_implementation == "python":
                self.compute_similarity_object = ComputeSimilarityPython(dataMatrix, **args)

            else:

                raise ValueError("ComputeSimilarity: value for argument 'use_implementation' not recognized")

    def compute_similarity(self, **args):

        return self.compute_similarity_object.compute_similarity(**args)
