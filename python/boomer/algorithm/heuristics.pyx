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

        According to the notation in http://www.ke.tu-darmstadt.de/bibtex/publications/show/3201, a confusion matrix
        consists of 8 elements, namely CIN, CIP, CRN, CRP, UIN, UIP, URN and URP. The individual symbols used in this
        notation have the following meaning:

        - The first symbol denotes whether the corresponding labels are covered (C) or uncovered (U) by the rule.
        - The second symbol denotes relevant (R) or irrelevant (I) labels according to the ground truth.
        - The third symbol denotes labels for which the prediction in the rule's head is positive (P) or negative (N).

        This results in the terminology given in the following table:

                   | ground-   |           |
                   | truth     | predicted |
        -----------|-----------|-----------|-----
         covered   |         0 |         0 | CIN
                   |         0 |         1 | CIP
                   |         1 |         0 | CRN
                   |         1 |         1 | CRP
        -----------|-----------|-----------|-----
         uncovered |         0 |         0 | UIN
                   |         0 |         1 | UIP
                   |         1 |         0 | URN
                   |         1 |         1 | URP

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
    A heuristic that measures the fraction of incorrectly predicted labels among all labels.

    It calculates as `(CIP + CRN + URN + URP) / (CIN + CIP + CRN + CRP + UIN + UIP + URN + URP)`, where the division by
    zero evaluates to 1, by definition.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_incorrect_labels = cip + crn + urn + urp
        cdef float64 num_total_labels = num_incorrect_labels + cin + crp + uin + uip

        if num_total_labels == 0:
            return 1

        return num_incorrect_labels / num_total_labels


cdef class Precision(Heuristic):
    """
    A heuristic that measures the fraction of incorrectly predicted labels among all covered labels.

    It calculates as `1 - ((CIN + CRP) / (CIN + CIP + CRN + CRP)) = (CIP + CRN) / (CIN + CIP + CRN + CRP)`, where the
    division by zero evaluates to 1, by definition.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_incorrect_labels = cip + crn
        cdef float64 num_covered_labels = num_incorrect_labels + cin + crp

        if num_covered_labels == 0:
            return 1

        return num_incorrect_labels / num_covered_labels


cdef class Recall(Heuristic):
    """
    A heuristic that measures the fraction of uncovered labels among all labels for which the rule's prediction is (or
    would be) correct, i.e., for which the ground truth is equal to the rule's prediction.

    It calculates as `1 - ((CIN + CRP) / (CIN + CRP + UIN + URP)) = (UIN + URP) / (CIN + CRP + UIN + URP)`, where the
    division by zero evaluates to 1, by definition.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_uncovered_equal = uin + urp
        cdef float64 num_equal = num_uncovered_equal + cin + crp

        if num_equal == 0:
            return 1

        return num_uncovered_equal / num_equal
