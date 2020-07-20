"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all classes that allow to store statistics about the labels of training examples.
"""


cdef class RefinementSearch:
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        """
        TODO

        :param example_index:
        :param weight:
        """
        pass

    cdef void reset_search(self):
        """
        TODO
        """
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        """
        TODO

        :param uncovered:
        :param accumulated:
        """
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        """
        TODO

        :param uncovered:
        :param accumulated:
        """
        pass


cdef class DecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics in the decomposable case, i.e., when the label-wise predictions are the same as the example-wise
    predictions.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        # In the decomposable case, the example-wise predictions are the same as the label-wise predictions...
        return <Prediction*>self.calculate_label_wise_prediction(uncovered, accumulated)


cdef class NonDecomposableRefinementSearch(RefinementSearch):
    """
    A base class for all classes that allow to search for the best refinement of a rule based on previously stored
    statistics in the non-decomposable case, i.e., when the label-wise predictions are not the same as the example-wise
    predictions.
    """

    cdef void update_search(self, intp example_index, uint32 weight):
        pass

    cdef void reset_search(self):
        pass

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        pass

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        pass


cdef class Statistics:
    """
    A base class for all classes that store statistics about the labels of the training examples.
    """

    cdef void reset_examples(self):
        """
        TODO
        """
        pass

    cdef void add_sampled_example(self, intp example_index, uint32 weight):
        """
        TODO

        :param example_index:
        :param weight:
        """
        pass

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        """
        TODO

        :param example_index:
        :param weight:
        :param remove:
        """
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        """
        TODO

        :param label_indices:
        :return:
        """
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head):
        """
        TODO

        :param example_index:
        :param label_indices:
        :param head:
        """
        pass
