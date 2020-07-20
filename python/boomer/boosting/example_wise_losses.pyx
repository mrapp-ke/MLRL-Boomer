"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement loss functions that are applied example-wise.
"""
from boomer.common._arrays cimport uint8, array_float64, c_matrix_float64, fortran_matrix_float64, get_index
from boomer.boosting.differentiable_losses cimport _l2_norm_pow

from libc.math cimport pow, fabs
from libc.stdlib cimport malloc, free

from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv


cdef class ExampleWiseRefinementSearch(NonDecomposableRefinementSearch):
    """
    Allows to search for the best refinement of a rule according to a differentiable loss function that is applied
    example-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight, const intp[::1] label_indices,
                  const float64[:, ::1] gradients, const float64[::1] total_sums_of_gradients,
                  const float64[:, ::1] hessians, const float64[::1] total_sums_of_hessians):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            optimal scores to be predicted by rules
        :param label_indices:               An array of dtype int, shape `(num_considered_labels)`, representing the
                                            indices of the labels that should be considered by the search or None, if
                                            all labels should be considered
        :param gradients:                   An array of dtype float, shape `(num_examples, num_labels)`, representing
                                            the gradient for each example and label
        :param total_sums_of_gradients:     An array of dtype float, shape `(num_labels)`, representing the sum of the
                                            gradients of all examples, which should be considered by the search, for
                                            each label
        :param hessians:                    An array of dtype float, shape `(num_examples, num_hessians)`, representing
                                            the hessian for each example and label
        :param total_sums_of_hessians:      An array of dtype float, shape `(num_hessians)`, representing the sum of the
                                            hessians of all examples, which should be considered by the search, for each
                                            label
        """
        self.l2_regularization_weight = l2_regularization_weight
        self.label_indices = label_indices
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients
        cdef intp num_gradients = gradients.shape[1] if label_indices is None else label_indices.shape[0]
        cdef float64[::1] sums_of_gradients = array_float64(num_gradients)
        sums_of_gradients[:] = 0
        self.sums_of_gradients = sums_of_gradients
        self.accumulated_sums_of_gradients = None
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians
        cdef intp num_hessians = __triangular_number(num_gradients)
        cdef float64[::1] sums_of_hessians = array_float64(num_hessians)
        sums_of_hessians[:] = 0
        self.sums_of_hessians = sums_of_hessians
        self.accumulated_sums_of_hessians = None
        cdef LabelWisePrediction* prediction = new LabelWisePrediction(num_gradients, NULL, NULL, 0)
        self.prediction = prediction

    def __dealloc__(self):
        del self.prediction

    cdef void update_search(self, intp example_index, uint32 weight):
        # Class members
        cdef const float64[:, ::1] gradients = self.gradients
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef const float64[:, ::1] hessians = self.hessians
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        cdef const intp[::1] label_indices = self.label_indices
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # Temporary variables
        cdef intp i, c, c2, l, l2, offset

        # Add the gradients and hessians of the example at the given index (weighted by the given weight) to the current
        # sum of gradients and hessians...
        i = 0

        for c in range(num_gradients):
            l = get_index(c, label_indices)
            sums_of_gradients[c] += (weight * gradients[example_index, l])
            offset = __triangular_number(l)

            for c2 in range(c + 1):
                l2 = offset + get_index(c2, label_indices)
                sums_of_hessians[i] += (weight * hessians[example_index, l2])
                i += 1

    cdef void reset_search(self):
        # Class members
        cdef float64[::1] sums_of_gradients = self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.sums_of_hessians
        # The number of gradients
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # The number of hessians
        cdef intp num_hessians = sums_of_hessians.shape[0]
        # Temporary variables
        cdef intp c
        # Update the arrays that store the accumulated sums of gradients and hessians...
        cdef float64[::1] accumulated_sums_of_gradients = self.accumulated_sums_of_gradients
        cdef float64[::1] accumulated_sums_of_hessians

        if accumulated_sums_of_gradients is None:
            accumulated_sums_of_gradients = array_float64(num_gradients)
            self.accumulated_sums_of_gradients = accumulated_sums_of_gradients
            accumulated_sums_of_hessians = array_float64(num_hessians)
            self.accumulated_sums_of_hessians = accumulated_sums_of_hessians

            for c in range(num_gradients):
                accumulated_sums_of_gradients[c] = sums_of_gradients[c]
                sums_of_gradients[c] = 0

            for c in range(num_hessians):
                accumulated_sums_of_hessians[c] = sums_of_hessians[c]
                sums_of_hessians[c] = 0
        else:
            accumulated_sums_of_hessians = self.accumulated_sums_of_hessians

            for c in range(num_gradients):
                accumulated_sums_of_gradients[c] += sums_of_gradients[c]
                sums_of_gradients[c] = 0

            for c in range(num_hessians):
                accumulated_sums_of_hessians[c] += sums_of_hessians[c]
                sums_of_hessians[c] = 0

    cdef LabelWisePrediction* calculate_label_wise_prediction(self, bint uncovered, bint accumulated):
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef LabelWisePrediction* prediction = self.prediction
        cdef intp num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64* quality_scores = prediction.qualityScores_
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]

        # To avoid array recreation each time this function is called, the arrays for storing predictions and quality
        # scores are only (re-)initialized if they have not been initialized yet, or if they have the wrong size.
        if predicted_scores == NULL or num_predictions != num_gradients:
            predicted_scores = <float64*>malloc(num_gradients * sizeof(float64))
            prediction.predictedScores_ = predicted_scores
            quality_scores = <float64*>malloc(num_gradients * sizeof(float64))
            prediction.qualityScores_ = quality_scores

        # The overall quality score, i.e. the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef const float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef const intp[::1] label_indices
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, c2, l, l2

        if uncovered:
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            label_indices = self.label_indices

        # For each label, calculate the score to be predicted, as well as a quality score...
        for c in range(num_gradients):
            sum_of_gradients = sums_of_gradients[c]
            c2 = __triangular_number(c + 1) - 1
            sum_of_hessians = sums_of_hessians[c2]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                l2 = __triangular_number(l + 1) - 1
                sum_of_hessians = total_sums_of_hessians[l2] - sum_of_hessians

            # Calculate score to be predicted for the current label...
            score = sum_of_hessians + l2_regularization_weight
            score = -sum_of_gradients / score if score != 0 else 0
            predicted_scores[c] = score

            # Calculate the quality score for the current label...
            score_pow = pow(score, 2)
            score = (sum_of_gradients * score) + (0.5 * score_pow * sum_of_hessians)
            quality_scores[c] = score + (0.5 * l2_regularization_weight * score_pow)
            overall_quality_score += score

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores, num_gradients)
        prediction.overallQualityScore_ = overall_quality_score

        return prediction

    cdef Prediction* calculate_example_wise_prediction(self, bint uncovered, bint accumulated):
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef Prediction* prediction = <Prediction*>self.prediction
        cdef float64[::1] sums_of_gradients = self.accumulated_sums_of_gradients if accumulated else self.sums_of_gradients
        cdef float64[::1] sums_of_hessians = self.accumulated_sums_of_hessians if accumulated else self.sums_of_hessians
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # Temporary variables
        cdef const float64[::1] total_sums_of_gradients, total_sums_of_hessians
        cdef float64[::1] gradients, hessians,
        cdef const intp[::1] label_indices
        cdef intp num_hessians, c, c2, l, l2, i, offset

        if uncovered:
            label_indices = self.label_indices
            num_hessians = sums_of_hessians.shape[0]
            gradients = array_float64(num_gradients)
            hessians = array_float64(num_hessians)
            total_sums_of_gradients = self.total_sums_of_gradients
            total_sums_of_hessians = self.total_sums_of_hessians
            i = 0

            for c in range(num_gradients):
                l = get_index(c, label_indices)
                gradients[c] = total_sums_of_gradients[l] - sums_of_gradients[c]
                offset = __triangular_number(l)

                for c2 in range(c + 1):
                    l2 = offset + get_index(c2, label_indices)
                    hessians[i] = total_sums_of_hessians[l2] - sums_of_hessians[i]
                    i += 1
        else:
            gradients = sums_of_gradients
            hessians = sums_of_hessians

        # Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
        cdef float64* predicted_scores = __dsysv_float64(hessians, gradients, l2_regularization_weight)
        prediction.predictedScores_ = predicted_scores

        # Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
        cdef float64 overall_quality_score = __ddot_float64(predicted_scores, &gradients[0], num_gradients)
        cdef float64* tmp = __dspmv_float64(&hessians[0], predicted_scores, num_gradients)
        overall_quality_score += 0.5 * __ddot_float64(predicted_scores, tmp, num_gradients)

        # Add the L2 regularization term to the overall quality score...
        overall_quality_score += 0.5 * l2_regularization_weight * _l2_norm_pow(predicted_scores, num_gradients)
        prediction.overallQualityScore_ = overall_quality_score

        return prediction


cdef class ExampleWiseLoss(DifferentiableLoss):
    """
    Allows to locally minimize a differentiable (surrogate) loss function that is applied example-wise by the rules that
    are learned by a boosting algorithm.
    """

    def __cinit__(self, ExampleWiseLossFunction loss_function, float64 l2_regularization_weight):
        """
        :param loss_function:               An example-wise loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            optimal scores to be predicted by rules. Increasing this value causes the
                                            model to be more conservative, setting it to 0 turns of L2 regularization
                                            entirely
        """
        self.loss_function = loss_function
        self.l2_regularization_weight = l2_regularization_weight

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        # An example-wise loss function to be minimized
        cdef ExampleWiseLossFunction loss_function = self.loss_function
        # The weight to be used for L2 regularization
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of labels
        cdef intp num_labels = label_matrix.num_labels
        # The number of hessians
        cdef intp num_hessians = __triangular_number(num_labels)
        # A matrix that stores the gradients for each example
        cdef float64[:, ::1] gradients = c_matrix_float64(num_examples, num_labels)
        # An array that stores the column-wise sums of the matrix of gradients
        cdef float64[::1] total_sums_of_gradients = array_float64(num_labels)
        total_sums_of_gradients[:] = 0
        # A matrix that stores the hessians for each example
        cdef float64[:, ::1] hessians = c_matrix_float64(num_examples, num_hessians)
        # An array that stores the column-wise sums of the matrix of hessians
        cdef float64[::1] total_sums_of_hessians = array_float64(num_hessians)
        total_sums_of_hessians[:] = 0
        # A matrix that stores the currently predicted scores for each example and label
        cdef float64[:, ::1] current_scores = c_matrix_float64(num_examples, num_labels)
        # An array that stores the scores that are predicted by the default rule
        cdef float64* predicted_scores = <float64*>malloc(num_labels * sizeof(float64))
        # Temporary variables
        cdef intp r, c

        for c in range(num_labels):
            predicted_scores[c] = 0

        # Traverse each example to calculate the initial gradients and hessians...
        for r in range(num_examples):
            loss_function.calculate_gradients_and_hessians(label_matrix, r, predicted_scores, gradients[r, :],
                                                           hessians[r, :])

            for c in range(num_labels):
                total_sums_of_gradients[c] += gradients[r, c]

            for c in range(num_hessians):
                total_sums_of_hessians[c] += hessians[r, c]

        # Compute the optimal scores to be predicted by the default rule by solving the system of linear equations...
        predicted_scores = __dsysv_float64(total_sums_of_hessians, total_sums_of_gradients, l2_regularization_weight)

        # Traverse each example again to calculate the updated gradients and hessians based on the calculated scores...
        for r in range(num_examples):
            for c in range(num_labels):
                current_scores[r, c] = predicted_scores[c]

            loss_function.calculate_gradients_and_hessians(label_matrix, r, predicted_scores, gradients[r, :],
                                                           hessians[r, :])

        # Store the gradients...
        self.gradients = gradients
        self.total_sums_of_gradients = total_sums_of_gradients

        # Store the hessians...
        self.hessians = hessians
        self.total_sums_of_hessians = total_sums_of_hessians

        # Store the label matrix and the currently predicted scores...
        self.label_matrix = label_matrix
        self.current_scores = current_scores

        return new DefaultPrediction(num_labels, predicted_scores)

    cdef void reset_examples(self):
        # Class members
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # Reset total sums of gradients and hessians to 0...
        total_sums_of_gradients[:] = 0
        total_sums_of_hessians[:] = 0

    cdef void update_covered_example(self, intp example_index, uint32 weight, bint remove):
        # Class members
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        # The number of gradients/hessians...
        cdef intp num_elements = gradients.shape[1]
        # The given weight multiplied by 1 or -1, depending on the argument `remove`
        cdef float64 signed_weight = -<float64>weight if remove else weight
        # Temporary variables
        cdef intp c

        # For each label, add the gradient of the example at the given index (weighted by the given weight) to the total
        # sums of gradients...
        for c in range(num_elements):
            total_sums_of_gradients[c] += (signed_weight * gradients[example_index, c])

        # Add the hessians of the example at the given index (weighted by the given weight) to the total sums of
        # hessians...
        num_elements = hessians.shape[1]

        for c in range(num_elements):
            total_sums_of_hessians[c] += (signed_weight * hessians[example_index, c])

    cdef RefinementSearch begin_search(self, intp[::1] label_indices):
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[::1] total_sums_of_gradients = self.total_sums_of_gradients
        cdef float64[:, ::1] hessians = self.hessians
        cdef float64[::1] total_sums_of_hessians = self.total_sums_of_hessians
        return ExampleWiseRefinementSearch.__new__(ExampleWiseRefinementSearch, l2_regularization_weight, label_indices,
                                                   gradients, total_sums_of_gradients, hessians, total_sums_of_hessians)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, HeadCandidate* head):
        # Class members
        cdef ExampleWiseLossFunction loss_function = self.loss_function
        cdef LabelMatrix label_matrix = self.label_matrix
        cdef float64[:, ::1] current_scores = self.current_scores
        cdef float64[:, ::1] gradients = self.gradients
        cdef float64[:, ::1] hessians = self.hessians
        # The number of predicted labels
        cdef intp num_predictions = head.numPredictions_
        # The predicted scores
        cdef float64* predicted_scores = head.predictedScores_
        # Temporary variables
        cdef intp c, l

        # Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
        # example at the given index...
        for c in range(num_predictions):
            l = get_index(c, label_indices)
            current_scores[example_index, l] += predicted_scores[c]

        # Update the gradients and hessians for the example at the given index...
        loss_function.calculate_gradients_and_hessians(label_matrix, example_index,
                                                       &current_scores[example_index, :][0],
                                                       gradients[example_index, :], hessians[example_index, :])


cdef inline intp __triangular_number(intp n):
    """
    Computes and returns the n-th triangular number, i.e., the number of elements in a n times n triangle.

    :param n:   A scalar of dtype `intp`, representing the order of the triangular number
    :return:    A scalar of dtype `intp`, representing the n-th triangular number
    """
    return (n * (n + 1)) // 2


cdef inline float64 __ddot_float64(float64* x, float64* y, int n):
    """
    Computes and returns the dot product x * y of two vectors using BLAS' DDOT routine (see
    http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga75066c4825cb6ff1c8ec4403ef8c843a.html).

    :param x:   A pointer to an array of type `float64`, shape (n), representing the first vector x
    :param y:   A pointer to an array of type `float64`, shape (n), representing the second vector y
    :param n:   The number of elements in the arrays `x` and `y`
    :return:    A scalar of dtype `float64`, representing the result of the dot product x * y
    """
    # Storage spacing between the elements of the arrays x and y
    cdef int inc = 1
    # Invoke the DDOT routine...
    cdef float64 result = ddot(&n, x, &inc, y, &inc)
    return result


cdef inline float64* __dspmv_float64(float64* a, float64* x, int n):
    """
    Computes and returns the solution to the matrix-vector operation A * x using BLAS' DSPMV routine (see
    http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gab746575c4f7dd4eec72e8110d42cefe9.html).
    This function expects A to be a double-precision symmetric matrix with shape `(n, n)` and x to be a double-precision
    array with shape `(n)`.

    DSPMV expects the matrix A to be supplied in packed form, i.e., as an array with shape `(n * (n + 1) // 2 )` that
    consists of the columns of A appended to each other and omitting all unspecified elements.

    :param a:   A pointer to an array of type `float64`, shape `(n * (n + 1) // 2)`, representing the elements in the
                upper-right triangle of the matrix A in a packed form
    :param x:   A pointer to an array of type `float64`, shape `(n)`, representing the elements in the array x
    :param n:   The number of elements in the arrays `a` and `x`
    :return:    A pointer to an array of type `float64`, shape `(n)`, representing the result of the matrix-vector
                operation A * x
    """
    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # A scalar to be multiplied with the matrix A
    cdef float64 alpha = 1
    # The increment for the elements of x
    cdef int incx = 1
    # A scalar to be multiplied with vector y
    cdef float64 beta = 0
    # An array of type `float64`, shape `(n)`. Will contain the result of A * x
    cdef float64* y = <float64*>malloc(n * sizeof(float64))
    # The increment for the elements of y
    cdef int incy = 1
    # Invoke the DSPMV routine...
    dspmv(uplo, &n, &alpha, a, x, &incx, &beta, y, &incy)
    return y


cdef inline float64* __dsysv_float64(float64[::1] coefficients, float64[::1] inverted_ordinates,
                                     float64 l2_regularization_weight):
    """
    Computes and returns the solution to a system of linear equations A * X = B using LAPACK's DSYSV solver (see
    http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
    DSYSV requires A to be a double-precision matrix with shape `(num_equations, num_equations)`, representing the
    coefficients, and B to be a double-precision matrix with shape `(num_equations, nrhs)`, representing the ordinates.
    X is a matrix of unknowns with shape `(num_equations, nrhs)`.

    DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to the system
    of linear equations. To retain their state, this function will copy the given arrays before invoking DSYSV.

    Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the upper-right
    triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency, this function expects
    the coefficients to be given as an array with shape `num_equations * (num_equations + 1) // 2`, representing the
    elements of the upper-right triangle of A, where the columns are appended to each other and unspecified elements are
    omitted. This function will implicitly convert the given array into a matrix that is suited for DSYSV.

    Optionally, this function allows to specify a weight to be used for L2 regularization. The given weight is added to
    each element on the diagonal of the matrix of coefficients A.

    :param coefficients:                An array of dtype `float64`, shape `num_equations * (num_equations + 1) // 2)`,
                                        representing coefficients
    :param inverted_ordinates:          An array of dtype `float64`, shape `(num_equations)`, representing the inverted
                                        ordinates, i.e., ordinates * -1. The sign of the elements in this array will be
                                        inverted to when creating the matrix B
    :param l2_regularization_weight:    A scalar of dtype `float64`, representing the weight of the L2 regularization
    :return:                            A pointer to an array of type `float64`, shape `(num_equations)`, representing
                                        the solution to the system of linear equations
    """
    cdef float64[::1] result
    cdef float64 tmp
    cdef intp r, c, i
    # The number of linear equations
    cdef int n = inverted_ordinates.shape[0]
    # Create the array A by copying the array `coefficients`. DSYSV requires the array A to be Fortran-contiguous...
    cdef float64[::1, :] a = fortran_matrix_float64(n, n)
    i = 0

    for c in range(n):
        for r in range(c + 1):
            tmp = coefficients[i]

            if r == c:
                tmp += l2_regularization_weight

            a[r, c] = tmp
            i += 1

    # Create the array B by copying the array `inverted_ordinates` and inverting its elements. It will be overwritten
    # with the solution to the system of linear equations. DSYSV requires the array B to be Fortran-contiguous...
    cdef float64* b = <float64*>malloc(n * sizeof(float64))

    for r in range(n):
        b[r] = -inverted_ordinates[r]

    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of right-hand sides, i.e, the number of columns of the matrix B
    cdef int nrhs = 1
    # Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    cdef int info
    # We must query optimal value for the argument `lwork` (the length of the working array `work`)...
    cdef double worksize
    cdef int lwork = -1  # -1 means that the optimal value should be queried
    dsysv(uplo, &n, &nrhs, &a[0, 0], &n, <int*>0, &b[0], &n, &worksize, &lwork, &info)  # Queries the optimal value
    lwork = <int>worksize
    # Allocate the working array...
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    # Allocate another working array...
    cdef int* ipiv = <int*>malloc(n * sizeof(int))

    try:
        # Run the DSYSV solver...
        dsysv(uplo, &n, &nrhs, &a[0, 0], &n, ipiv, &b[0], &n, work, &lwork, &info)

        if info == 0:
            # The solution has been computed successfully...
            return b
        else:
            # An error occurred...
            raise ArithmeticError('DSYSV terminated with non-zero info code: ' + str(info))
    finally:
        # Free the allocated memory...
        free(<void*>ipiv)
        free(<void*>work)
