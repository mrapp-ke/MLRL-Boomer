"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.common._arrays cimport array_float64, fortran_matrix_float64, get_index
from boomer.boosting._math cimport l2_norm_pow, triangular_number

from libc.stdlib cimport malloc, free

from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv


cdef class ExampleWiseDefaultRuleEvaluation(DefaultRuleEvaluation):
    """
    Allows to calculate the predictions of a default rule such that they minimize a loss function that is applied
    example-wise.
    """

    def __cinit__(self, ExampleWiseLossFunction loss_function, float64, l2_regularization_weight):
        """
        :param loss_function:               The loss function to be minimized
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by the default rule
        """
        self.loss_function = loss_function
        self.l2_regularization_weight = l2_regularization_weight

    cdef DefaultPrediction* calculate_default_prediction(self, LabelMatrix label_matrix):
        # Class members
        cdef ExampleWiseLossFunction loss_function = self.loss_function
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of examples
        cdef intp num_examples = label_matrix.num_examples
        # The number of gradients
        cdef intp num_gradients = label_matrix.num_labels
        # The number of Hessians
        cdef intp num_hessians = triangular_number(num_gradients)
        # An array that stores the gradients for an example
        cdef float64[::1] gradients = array_float64(num_gradients)
        # An array that stores the sum of gradients
        cdef float64[::1] sums_of_gradients = array_float64(num_gradients)
        sums_of_gradients[:] = 0
        # An array that stores the Hessians for an example
        cdef float64[::1] hessians = array_float64(num_hessians)
        # An array that stores the sum of Hessians
        cdef float64[::1] sums_of_hessians = array_float64(num_hessians)
        sums_of_hessians[:] = 0
        # An array of zeros that represents the initially predicted scores
        cdef float64[::1] default_predictions = array_float64(num_gradients)
        default_predictions[:] = 0
        # Temporary variables
        cdef intp r, c

        for r in range(num_examples):
            # Calculate the gradients and Hessians for the current example...
            loss_function.calculate_gradients_and_hessians(label_matrix, r, &default_predictions[0], gradients,
                                                           hessians)

            for c in range(num_gradients):
                sums_of_gradients[c] += gradients[c]

            for c in range(num_hessians):
                sums_of_hessians[c] += hessians[c]

        # Calculate the scores to be predicted by the default rule by solving the system of linear equations...
        cdef float64* predicted_scores = __dsysv_float64(sums_of_hessians, sums_of_gradients, l2_regularization_weight)
        return new DefaultPrediction(num_gradients, predicted_scores)


cdef class ExampleWiseRuleEvaluation:
    """
    Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they minimize a
    loss function that is applied example-wise.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.l2_regularization_weight = l2_regularization_weight

    cdef void calculate_label_wise_prediction(self, const intp[::1] label_indices,
                                              const float64[::1] total_sums_of_gradients,
                                              float64[::1] sums_of_gradients, const float64[::1] total_sums_of_hessians,
                                              float64[::1] sums_of_hessians, bint uncovered,
                                              LabelWisePrediction* prediction):
        """
        Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
        label-wise sums of gradients and Hessians that are covered by the rule. The predicted scores and quality scores
        are stored in a given object of type `LabelWisePrediction`.

        If the argument `uncovered` is 1, the rule is considered to cover the difference between the sums of gradients
        and Hessians that are stored in the arrays `total_sums_of_gradients` and `sums_of_gradients` and
        `total_sums_of_hessians` and `sums_of_hessians`, respectively.

        :param label_indices:           An array of dtype `intp`, shape `(num_gradients)`, representing the indices of
                                        the labels for which the rule should predict or None, if the rule should predict
                                        for all labels
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(num_gradients), representing the total sums
                                        of gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(num_gradients)`, representing the sums of
                                        gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the total sums of
                                        Hessians for individual labels
        :param sums_of_hessians:        An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the sums of
                                        Hessians for individual labels
        :param uncovered:               0, if the rule covers the sums of gradient and Hessians that are stored in the
                                        array `sums_of_gradients` and `sums_of_hessians`, 1, if the rule covers the
                                        difference between the sums of gradients and Hessians that are stored in the
                                        arrays `total_sums_of_gradients` and `sums_of_gradients` and
                                        `total_sums_of_hessians` and `sums_of_hessians`, respectively.
        :param prediction:              A pointer to an object of type `LabelWisePrediction` that should be used to
                                        store the predicted scores and quality scores
        """
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # The number of elements in the arrays `predicted_scores` and `quality_scores`
        cdef intp num_predictions = prediction.numPredictions_
        # The array that should be used to store the predicted scores
        cdef float64* predicted_scores = prediction.predictedScores_
        # The array that should be used to store the quality scores
        cdef float64* quality_scores = prediction.qualityScores_
        # The overall quality score, i.e. the sum of the quality scores for each label plus the L2 regularization term
        cdef float64 overall_quality_score = 0
        # Temporary variables
        cdef float64 sum_of_gradients, sum_of_hessians, score, score_pow
        cdef intp c, c2, l, l2

        # To avoid array recreation each time this function is called, the arrays for storing predictions and quality
        # scores are only (re-)initialized if they have not been initialized yet, or if they have the wrong size.
        if predicted_scores == NULL or num_predictions != num_gradients:
            predicted_scores = <float64*>malloc(num_gradients * sizeof(float64))
            prediction.predictedScores_ = predicted_scores
            quality_scores = <float64*>malloc(num_gradients * sizeof(float64))
            prediction.qualityScores_ = quality_scores

        # For each label, calculate the score to be predicted, as well as a quality score...
        for c in range(num_gradients):
            sum_of_gradients = sums_of_gradients[c]
            c2 = triangular_number(c + 1) - 1
            sum_of_hessians = sums_of_hessians[c2]

            if uncovered:
                l = get_index(c, label_indices)
                sum_of_gradients = total_sums_of_gradients[l] - sum_of_gradients
                l2 = triangular_number(l + 1) - 1
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
        overall_quality_score += 0.5 * l2_regularization_weight * l2_norm_pow(predicted_scores, num_gradients)
        prediction.overallQualityScore_ = overall_quality_score

    cdef void calculate_example_wise_prediction(self, const intp[::1] label_indices,
                                                const float64[::1] total_sums_of_gradients,
                                                float64[::1] sums_of_gradients,
                                                const float64[::1] total_sums_of_hessians,
                                                float64[::1] sums_of_hessians, bint uncovered,
                                                Prediction* prediction):
        """
        Calculates the scores to be predicted by a rule, as well as an overall quality score, based on the sums of
        gradients and Hessians that are covered by the rule. The predicted scores and quality scores are stored in a
        given object of type `Prediction`.

        If the argument `uncovered` is 1, the rule is considered to cover the difference between the sums of gradients
        and Hessians that are stored in the arrays `total_sums_of_gradients` and `sums_of_gradients` and
        `total_sums_of_hessians` and `sums_of_hessians`, respectively.

        :param label_indices:           An array of dtype `intp`, shape `(num_gradients)`, representing the indices of
                                        the labels for which the rule should predict or None, if the rule should predict
                                        for all labels
        :param total_sums_of_gradients: An array of dtype `float64`, shape `(num_gradients), representing the total sums
                                        of gradients for individual labels
        :param sums_of_gradients:       An array of dtype `float64`, shape `(num_gradients)`, representing the sums of
                                        gradients for individual labels
        :param total_sums_of_hessians:  An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the total sums of
                                        Hessians for individual labels
        :param sums_of_hessians:        An array of dtype `float64`, shape
                                        `((num_gradients + (num_gradients + 1)) // 2)`, representing the sums of
                                        Hessians for individual labels
        :param uncovered:               0, if the rule covers the sums of gradient and Hessians that are stored in the
                                        array `sums_of_gradients` and `sums_of_hessians`, 1, if the rule covers the
                                        difference between the sums of gradients and Hessians that are stored in the
                                        arrays `total_sums_of_gradients` and `sums_of_gradients` and
                                        `total_sums_of_hessians` and `sums_of_hessians`, respectively.
        :param prediction:              A pointer to an object of type `Prediction` that should be used to store the
                                        predicted scores and quality score
        """
        # Class members
        cdef float64 l2_regularization_weight = self.l2_regularization_weight
        # The number of gradients considered by the current search
        cdef intp num_gradients = sums_of_gradients.shape[0]
        # Temporary variables
        cdef float64[::1] gradients, hessians
        cdef intp num_hessians, c, c2, l, l2, i, offset

        if uncovered:
            num_hessians = sums_of_hessians.shape[0]
            gradients = array_float64(num_gradients)
            hessians = array_float64(num_hessians)
            i = 0

            for c in range(num_gradients):
                l = get_index(c, label_indices)
                gradients[c] = total_sums_of_gradients[l] - sums_of_gradients[c]
                offset = triangular_number(l)

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
        overall_quality_score += 0.5 * l2_regularization_weight * l2_norm_pow(predicted_scores, num_gradients)
        prediction.overallQualityScore_ = overall_quality_score


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
        free(ipiv)
        free(work)
