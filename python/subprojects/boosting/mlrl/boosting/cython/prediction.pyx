"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class ExampleWiseBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts known label vectors for given query examples by comparing the
    predicted regression scores or probability estimates to the label vectors encountered in the training data.
    """

    def is_based_on_probabilities(self) -> bool:
        """
        Returns whether binary predictions are derived from probability estimates rather than regression scores or not.

        :return: True, if binary predictions are derived from probability estimates rather than regression scores, False
                 otherwise
        """
        return self.config_ptr.isBasedOnProbabilities()

    def set_based_on_probabilities(self, based_on_probabilities: bool) -> ExampleWiseBinaryPredictorConfig:
        """
        Sets whether binary predictions should be derived from probability estimates rather than regression scores or
        not.

        :param based_on_probabilities:  True, if binary predictions should be derived from probability estimates rather
                                        than regression scores, False otherwise
        :return:                        An `ExampleWiseBinaryPredictorConfig` that allows further configuration of the
                                        predictor
        """
        self.config_ptr.setBasedOnProbabilities(based_on_probabilities)
        return self


cdef class GfmBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
    according to the general F-measure maximizer (GFM).
    """

    def is_based_on_probabilities(self) -> bool:
        """
        Returns whether binary predictions are derived from probability estimates rather than regression scores or not.

        :return: True, if binary predictions are derived from probability estimates rather than regression scores, False
                 otherwise
        """
        return self.config_ptr.isBasedOnProbabilities()

    def set_based_on_probabilities(self, based_on_probabilities: bool) -> GfmBinaryPredictorConfig:
        """
        Sets whether binary predictions should be derived from probability estimates rather than regression scores or
        not.

        :param based_on_probabilities:  True, if binary predictions should be derived from probability estimates rather
                                        than regression scores, False otherwise
        :return:                        An `GfmBinaryPredictorConfig` that allows further configuration of the predictor
        """
        self.config_ptr.setBasedOnProbabilities(based_on_probabilities)
        return self


cdef class LabelWiseBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
    individually.
    """

    def is_based_on_probabilities(self) -> bool:
        """
        Returns whether binary predictions are derived from probability estimates rather than regression scores or not.

        :return: True, if binary predictions are derived from probability estimates rather than regression scores, False
                 otherwise
        """
        return self.config_ptr.isBasedOnProbabilities()

    def set_based_on_probabilities(self, based_on_probabilities: bool) -> LabelWiseBinaryPredictorConfig:
        """
        Sets whether binary predictions should be derived from probability estimates rather than regression scores or
        not.

        :param based_on_probabilities:  True, if binary predictions should be derived from probability estimates rather
                                        than regression scores, False otherwise
        :return:                        A `LabelWiseBinaryPredictorConfig` that allows further configuration of the
                                        predictor
        """
        self.config_ptr.setBasedOnProbabilities(based_on_probabilities)
        return self
