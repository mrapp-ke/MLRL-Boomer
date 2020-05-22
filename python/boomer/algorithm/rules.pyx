"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides model classes that are used to build rule-based models.
"""
import numpy as np


cdef class Body:
    """
    A base class for the body of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[:] example):
        """
        Returns whether a certain example is covered by the body, or not.

        :param example: An array of dtype float, shape `(num_features)`, representing the features of an example
        :return:        1, if the example is covered, 0 otherwise
        """
        pass


cdef class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[:] example):
        return True


cdef class ConjunctiveBody(Body):
    """
    A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions and = or !=
    for nominal conditions, respectively.
    """

    def __cinit__(self, intp[::1] leq_feature_indices, float32[::1] leq_thresholds, intp[::1] gr_feature_indices,
                  float32[::1] gr_thresholds, intp[::1] eq_feature_indices, float32[::1] eq_thresholds,
                  intp[::1] neq_feature_indices, float32[::1] neq_thresholds):
        """
        :param leq_feature_indices: An array of dtype int, shape `(num_leq_conditions)`, representing the indices of the
                                    features, the numerical conditions that use the <= operator correspond to or None,
                                    if the body does not contain such a condition
        :param leq_thresholds:      An array of dtype float, shape `(num_leq_condition)`, representing the thresholds of
                                    the numerical conditions that use the <= operator or None, if the body does not
                                    contain such a condition
        :param gr_feature_indices:  An array of dtype int, shape `(num_gr_conditions)`, representing the indices of the
                                    features, the numerical conditions that use the > operator correspond to or None, if
                                    the body does not contain such a condition
        :param gr_thresholds:       An array of dtype float, shape `(num_gr_conditions)`, representing the thresholds of
                                    the numerical conditions that use the > operator or None, if the body does not
                                    contain such a condition
        :param eq_feature_indices:  An array of dtype int, shape `(num_eq_conditions)`, representing the indices of the
                                    features, the nominal conditions that use the = operator correspond to or None, if
                                    the body does not contain such a condition
        :param eq_thresholds:       An array of dtype float, shape `(num_eq_conditions)`, representing the thresholds of
                                    the nominal conditions that use the = operator or None, if the body does not contain
                                    such a condition
        :param neq_feature_indices: An array of dtype int, shape `(num_neq_conditions)`, representing the indices of the
                                    features, the nominal conditions that use the != operator correspond to or None, if
                                    the body does not contain such a condition
        :param neq_thresholds:      An array of dtype float, shape `(num_neq_conditions)`, representing the thresholds
                                    of the nominal conditions that use the != operator or None, if the body does not
                                    contain such a condition
        """
        self.leq_feature_indices = leq_feature_indices
        self.leq_thresholds = leq_thresholds
        self.gr_feature_indices = gr_feature_indices
        self.gr_thresholds = gr_thresholds
        self.eq_feature_indices = eq_feature_indices
        self.eq_thresholds = eq_thresholds
        self.neq_feature_indices = neq_feature_indices
        self.neq_thresholds = neq_thresholds

    def __getstate__(self):
        return (np.asarray(self.leq_feature_indices) if self.leq_feature_indices is not None else None,
                np.asarray(self.leq_thresholds) if self.leq_thresholds is not None else None,
                np.asarray(self.gr_feature_indices) if self.gr_feature_indices is not None else None,
                np.asarray(self.gr_thresholds) if self.gr_thresholds is not None else None,
                np.asarray(self.eq_feature_indices) if self.eq_feature_indices is not None else None,
                np.asarray(self.eq_thresholds) if self.eq_thresholds is not None else None,
                np.asarray(self.neq_feature_indices) if self.neq_feature_indices is not None else None,
                np.asarray(self.neq_thresholds) if self.neq_thresholds is not None else None)

    def __setstate__(self, state):
        self.leq_feature_indices = state[0]
        self.leq_thresholds = state[1]
        self.gr_feature_indices = state[2]
        self.gr_thresholds = state[3]
        self.eq_feature_indices = state[4]
        self.eq_thresholds = state[5]
        self.neq_feature_indices = state[6]
        self.neq_thresholds = state[7]

    cdef bint covers(self, float32[:] example):
        cdef intp[::1] feature_indices = self.leq_feature_indices
        cdef float32[::1] thresholds = self.leq_thresholds
        cdef intp num_conditions = feature_indices.shape[0]
        cdef intp i, c

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] > thresholds[i]:
                return False

        feature_indices = self.gr_feature_indices
        thresholds = self.gr_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] <= thresholds[i]:
                return False

        feature_indices = self.eq_feature_indices
        thresholds = self.eq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] != thresholds[i]:
                return False

        feature_indices = self.neq_feature_indices
        thresholds = self.neq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] == thresholds[i]:
                return False

        return True


cdef class Head:
    """
    A base class for the head of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted = None):
        """
        Applies the head's prediction to a given vector of predictions given that no prediction has yet been made.

        :param predicted:   An array of dtype float, shape `(num_labels)`, representing which labels have already been
                            predicted.
        :param predictions: An array of dtype float, shape `(num_labels)`, representing a vector of predictions
        """
        pass


cdef class FullHead(Head):
    """
    A full head that assigns a numerical score to each label.
    """

    def __cinit__(self, float64[::1] scores):
        """
        :param scores:  An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                        rule for each label
        """
        self.scores = scores

    def __getstate__(self):
        return np.asarray(self.scores)

    def __setstate__(self, state):
        scores = state
        self.scores = scores

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted = None):
        cdef float64[::1] scores = self.scores
        cdef intp num_cols = predictions.shape[0]
        cdef intp c

        for c in range(num_cols):
            if predicted is not None:
                if not predicted[c]:
                    predictions[c] = scores[c]
                    predicted[c] = 1
            else:
                predictions[c] += scores[c]


cdef class PartialHead(Head):
    """
    A partial head that assigns a numerical score to one or several labels.
    """

    def __cinit__(self, intp[::1] label_indices, float64[::1] scores):
        """
        :param label_indices:   An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the
                                labels for which the rule predicts
        :param scores:          An array of dtype float, shape `(num_predicted_labels)`, representing the scores that
                                are predicted by the rule
        """
        self.scores = scores
        self.label_indices = label_indices

    def __getstate__(self):
        return np.asarray(self.label_indices), np.asarray(self.scores)

    def __setstate__(self, state):
        label_indices, scores = state
        self.label_indices = label_indices
        self.scores = scores

    cdef void predict(self, float64[::1] predictions, intp[::1] predicted = None):
        cdef intp[::1] label_indices = self.label_indices
        cdef float64[::1] scores = self.scores
        cdef intp num_labels = label_indices.shape[0]
        cdef intp c, label

        for c in range(num_labels):
            label = label_indices[c]
            if predicted is not None:
                if not predicted[label]:
                    predictions[label] = scores[c]
                    predicted[label] = 1
            else:
                predictions[label] += scores[c]


cdef class Rule:
    """
    A rule consisting of a body and head.
    """

    def __cinit__(self, body: Body, head: Head):
        """
        :param body:    The body of the rule
        :param head:    The head of the rule
        """
        self.body = body
        self.head = head

    def __getstate__(self):
        return self.body, self.head

    def __setstate__(self, state):
        body, head = state
        self.body = body
        self.head = head

    cpdef predict(self, float32[::1, :] x, float64[:, ::1] predictions, intp[:, ::1] predicted = None):
        """
        Applies the rule's prediction to a matrix of predictions for all examples it covers.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the given examples
        :param predicted:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels
                                per example for which a prediction has already been made
        """
        cdef Body body = self.body
        cdef Head head = self.head
        cdef intp num_examples = x.shape[0]
        cdef intp r

        for r in range(num_examples):
            if body.covers(x[r, :]):
                if predicted is not None:
                    head.predict(predictions[r, :], predicted[r, :])
                else:
                    head.predict(predictions[r, :])
