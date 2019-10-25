# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
ctypedef np.int32_t int32
ctypedef np.float32_t float32
ctypedef np.float64_t float64

DTYPE_INDICES = np.int32

DTYPE_FEATURES = np.float32

DTYPE_SCORES = np.float64


cdef class Body:
    """
    A base class for the body of a rule.
    """

    cdef bint covers(self, float32[:] example):
        """
        Returns whether a certain example is covered by the body, or not.

        :param example: An array of dtype float, shape `(num_features)`, representing the features of an example
        """
        pass


cdef class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    cdef bint covers(self, float32[:] example):
        return 1


cdef class ConjunctiveBody(Body):
    """
    A body that given as a conjunction of numerical conditions using <= and > operators.
    """

    cdef int32[::1] leq_features

    cdef float32[::1] leq_thresholds

    cdef int32[::1] gr_features

    cdef float32[::1] gr_thresholds

    def __cinit__(self, int32[::1] leq_features, float32[::1] leq_thresholds, int32[::1] gr_features,
                  float32[::1] gr_thresholds):
        """
        :param leq_features:    An array of dtype int, shape `(num_leq_conditions)`, representing the features of the
                                conditions that use the <= operator
        :param leq_thresholds:  An array of dtype float, shape `(num_leq_condition)`, representing the thresholds of the
                                conditions that use the <= operator
        :param gr_features:     An array of dtype int, shape `(num_gr_conditions)`, representing the features of the
                                conditions that use the > operator
        :param gr_thresholds:   An array of dtype float, shape `(num_gr_conditions)`, representing the thresholds of the
                                conditions that use the > operator
        """
        self.leq_features = leq_features
        self.leq_thresholds = leq_thresholds
        self.gr_features = gr_features
        self.gr_thresholds = gr_thresholds

    cdef bint covers(self, float32[:] example):
        cdef int32[::1] leq_features = self.leq_features
        cdef float32[::1] leq_thresholds = self.leq_thresholds
        cdef int32[::1] gr_features = self.gr_features
        cdef float32[::1] gr_thresholds = self.gr_thresholds
        cdef Py_ssize_t num_leq_conditions = leq_features.shape[0]
        cdef Py_ssize_t num_gr_conditions = gr_features.shape[0]
        cdef Py_ssize_t i, c

        for i in range(num_leq_conditions):
            c = leq_features[i]

            if example[c] > leq_thresholds[i]:
                return 0

        for i in range(num_gr_conditions):
            c = gr_features[i]

            if example[c] <= gr_thresholds[i]:
                return 0

        return 1


cdef class Head:
    """
    A base class for the head of a rule.
    """

    cdef predict(self, float64[:] predictions):
        """
        Applies the head's prediction to a given vector of predictions.

        :param predictions: An array of dtype float, shape `(num_labels)`, representing a vector of predictions
        """
        pass


cdef class FullHead(Head):
    """
    A full head that assigns a numerical score to each label.
    """

    cdef readonly float64[::1] scores

    def __cinit__(self, float64[::1] scores):
        """
        :param scores:  An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                        rule for each label
        """
        self.scores = scores

    cdef predict(self, float64[:] predictions):
        cdef float64[::1] scores = self.scores
        cdef Py_ssize_t num_cols = predictions.shape[1]
        cdef Py_ssize_t c

        for c in range(num_cols):
            predictions[c] += scores[c]


cdef class PartialHead(Head):
    """
    A partial head that assigns a numerical score to one or several labels.
    """

    cdef readonly float64[::1] scores

    cdef readonly int32[::1] labels

    def __cinit__(self, float64[::1] scores, int32[::1] labels):
        """
        :param labels:  An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the labels
                        for which the rule predicts
        :param scores:  An array of dtype float, shape `(num_predicted_labels)`, representing the scores that are
                        predicted by the rule
                """
        self.scores = scores
        self.labels = labels

    cdef predict(self, float64[:] predictions):
        cdef int32[::1] labels = self.labels
        cdef float64[::1] scores = self.scores
        cdef Py_ssize_t num_labels = labels.shape[0]
        cdef Py_ssize_t c, label

        for c in range(num_labels):
            label = labels[c]
            predictions[label] += scores[c]


cdef class Rule:
    """
    A rule consisting of a body and head.
    """

    cdef readonly Body body

    cdef readonly Head head

    def __cinit__(self, body: Body, head: Head):
        """
        :param body:    The body of the rule
        :param head:    The head of the rule
        """
        self.body = body
        self.head = head

    cpdef predict(self, float32[::1, :] x, float64[::1, :] predictions):
        """
        Applies the rule's prediction to all examples it covers.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the given examples
        """
        cdef Body body = self.body
        cdef Head head = self.head
        cdef Py_ssize_t num_examples = x.shape[0]
        cdef Py_ssize_t c

        for r in range(num_examples):
            if body.covers(x[r, :]):
                head.predict(predictions[r, :])
