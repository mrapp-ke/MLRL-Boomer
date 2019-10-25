# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np

DTYPE_INDICES = np.intc

cdef class Body:
    """
    A base class for the body of a rule.
    """

    cpdef np.ndarray match(self, x: np.ndarray):
        pass


cdef class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    cpdef np.ndarray match(self, x: np.ndarray):
        cdef Py_ssize_t num_examples = x.shape[0]
        return np.full((num_examples), True, dtype=DTYPE_INDICES)


cdef class ConjunctiveBody(Body):
    """
    A body that given as a conjunction of numerical conditions using <= and > operators.
    """

    def __cinit__(self, leq_features: np.ndarray, leq_thresholds: np.ndarray, gr_features: np.ndarray,
                  gr_thresholds: np.ndarray):
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

    cpdef np.ndarray match(self, x: np.ndarray):
        return np.all(np.less_equal(x[:, self.leq_features], self.leq_thresholds), axis=1) & np.all(
            np.greater(x[:, self.gr_features], self.gr_thresholds), axis=1)


cdef class Head:
    """
    A base class for the head of a rule.
    """

    cpdef predict(self, predictions: np.ndarray):
        """
        Applies the head's prediction to a given matrix of predictions.

        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the corresponding examples
        """
        pass


cdef class FullHead(Head):
    """
    A full head that assigns a numerical score to each label.
    """

    cdef readonly np.ndarray scores

    def __cinit__(self, scores: np.ndarray):
        """
        :param scores:  An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                        rule for each label
        """
        self.scores = scores

    cpdef predict(self, predictions: np.ndarray):
        predictions += self.scores


cdef class PartialHead(Head):
    """
    A partial head that assigns a numerical score to one or several labels.
    """

    cdef readonly np.ndarray scores

    cdef readonly np.ndarray labels

    def __cinit__(self, scores: np.ndarray, labels: np.ndarray):
        """
        :param labels:  An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the labels
                        for which the rule predicts
        :param scores:  An array of dtype float, shape `(num_predicted_labels)`, representing the scores that are
                        predicted by the rule
                """
        self.scores = scores
        self.labels = labels

    cpdef predict(self, predictions: np.ndarray):
        predictions[:, self.labels] += self.scores


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

    cpdef predict(self, x: np.ndarray, predictions: np.ndarray):
        """
        Applies the rule's prediction to all examples it covers.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the given examples
        """
        self.head.predict(predictions[self.body.match(x), :])
