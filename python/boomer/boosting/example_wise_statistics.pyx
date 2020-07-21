"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable) loss
function that is applied example-wise.
"""
from boomer.common._arrays cimport array_float64, c_matrix_float64


cdef class ExampleWiseStatistics(GradientStatistics):
    """
    Allows to store gradients and Hessians that are calculated according to a loss function that is applied
    example-wise.
    """

    def __cinit__(self, ExampleWiseLossFunction loss_function):
        """
        :param loss_function: The loss function to be used for calculating gradients and Hessians
        """
        self.loss_function = loss_function

    cdef void apply_default_prediction(self, LabelMatrix label_matrix, DefaultPrediction* default_prediction):
        # Class members
        cdef ExampleWiseLossFunction loss_function = self.loss_function
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # The number of hessians
        cdef intp num_hessians = __triangular_number(num_labels)
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[:, ::1] current_scores = c_matrix_float64(num_examples, num_labels)
        # A matrix that stores the gradients for each example
        cdef float64[:, ::1] gradients = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        # A matrix that stores the Hessians for each example
        cdef float64[:, ::1] hessians = c_matrix_float64(num_examples, num_hessians)
        # An array that stores the column-wise sums of the matrix of Hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_hessians)
        # An array that stores the scores that are predicted by the default rule
        cdef float64* predicted_scores = default_prediction.predictedScores_ if default_prediction != NULL else NULL
        # Temporary variables
        cdef float64 predicted_score
        cdef intp r, c

        for r in range(num_examples):
            for c in range(num_labels):
                # Store the score that is predicted by the default rule for the current example and label...
                predicted_score = predicted_scores[c] if predicted_scores != NULL else 0
                current_scores[r, c] = predicted_scores[c]

            # Calculate the gradients and Hessians for the current example...
            loss_function.calculate_gradients_and_hessians(label_matrix, r, &current_scores[r, :][0], gradients[r, :],
                                                           hessians[r, :])

        # Store the gradients...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients

        # Store the Hessians...
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians

        # Store the label matrix and the currently predicted scores...
        self.label_matrix = label_matrix
        self.current_scores = current_scores

    cdef void reset_statistics(self):
        pass

    cdef void add_sampled_statistic(self, intp statistic_index, uint32 weight):
        pass

    cdef void update_covered_statistic(self, intp statistic_index, uint32 weight, bint remove):
        pass

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        pass

    cdef void apply_prediction(self, intp statistic_index, intp[::1] label_indices, HeadCandidate* head):
        pass


cdef inline intp __triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2
