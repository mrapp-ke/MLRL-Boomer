"""
Implements different heuristics for assessing the quality of single- or multi-label rules based on confusion matrices.
Given the elements of a confusion matrix, a heuristic calculates a quality score in [0, 1]. All heuristics must be
implemented as loss functions, i.e., rules with a smaller quality score are better than those with a large quality
score.
"""

cdef class Heuristic:
    """
    A base class for all heuristics.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        """
        Calculates and returns a quality score in [0, 1] given the elements of a confusion matrix.

        A confusion matrix consists of 8 elements, namely CIN, CIP, CRN, CRP, UIN, UIP, URN, URP. According to this
        notation, the individual symbols have the following meaning:

        - The first symbol denotes whether an element corresponds to labels that are covered (C) or uncovered (U) by the
          rule.
        - The second symbol denotes relevant (R) and irrelevant (I) labels according to the ground truth.
        - The third symbol denotes labels for which the prediction in the rule's head is positive (P) or negative (N).

        Real numbers may be used for the individual elements, if different weights are assigned to the corresponding
        labels.

        :param cin: The number of covered (C) labels that are irrelevant (I) according to the ground truth and for which
                    the prediction in the rule's head is negative (N)
        :param cip: The number of covered (C) labels that are irrelevant (I) according to the ground truth and for which
                    the prediction in the rule's head is positive (P)
        :param crn: The number of covered (C) labels that are relevant (R) according to the ground truth and for which
                    the prediction in the rule's head is negative (N)
        :param crp: The number of covered (C) labels that are relevant (R) according to the ground truth and for which
                    the prediction in the rule's head is positive (P)
        :param uin: The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and for
                    which the prediction in the rule's head is negative (N)
        :param uip: The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and for
                    which the prediction in the rule's head is positive (P)
        :param urn: The number of uncovered (U) labels that are relevant (R) according to the ground truth and for which
                    the prediction in the rule's head is negative (N)
        :param urp: The number of uncovered (U) labels that are relevant (R) according to the ground truth and for which
                    the prediction in the rule's head is positive (P)
        :return:    The quality score in that has been calculated
        """
        pass

cdef class HammingLoss(Heuristic):
    """
    A heuristic that calculates as the Hamming loss, i.e., as the fraction of correctly predicted labels among all
    labels.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_correct_labels = cip + crn + urn + urp
        cdef float64 num_total_labels = num_correct_labels + cin + crp + uin + uip

        if num_labels == 0:
            return 1
        return num_correct_labels / num_total_labels

cdef class Precision(Heuristic):
    """
    A heuristic that calculates as 1 - prec, where prec corresponds to the precision metric, i.e., as the fraction of
    incorrectly predicted labels among all covered labels.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_incorrect_labels = cip + crn
        cdef float64 num_covered_labels = num_incorrect_labels + cin + crp

        if num_covered_labels == 0:
            return 1
        return num_incorrect_labels / num_covered_labels
