"""
Implements different heuristics for assessing the quality of single- or multi-label rules based on confusion matrices.
Given the elements of a confusion matrix, a heuristic calculates a quality score in [0, 1]. All heuristics must be
implemented as loss functions, i.e., rules with a smaller quality score are better than those with a large quality
score.
"""
from libc.math cimport isinf, pow


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
        cdef float64 num_incorrect = cip + crn + urn + urp
        cdef float64 num_total = num_incorrect + cin + crp + uin + uip

        if num_total == 0:
            return 1

        return num_incorrect / num_total


cdef class Precision(Heuristic):
    """
    A heuristic that measures the fraction of incorrectly predicted labels among all covered labels.

    It calculates as `1 - ((CIN + CRP) / (CIN + CIP + CRN + CRP)) = (CIP + CRN) / (CIN + CIP + CRN + CRP)`, where the
    division by zero evaluates to 1, by definition.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_covered_incorrect = cip + crn
        cdef float64 num_covered = num_covered_incorrect + cin + crp

        if num_covered == 0:
            return 1

        return num_covered_incorrect / num_covered


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


cdef class WeightedRelativeAccuracy(Heuristic):
    """
    A heuristic that measures as the fraction of uncovered labels among all labels weighted by the difference between
    the fraction of covered labels and the fraction of labels for which the rule's prediction is (or would be) correct.

    It calculates as `1 - ((CIN + CIP + CRN + CRP) / (num_labels * (frac_covered - frac_equal))
    = (UIN + UIP + URN + URP) / (num_labels * (frac_covered - frac_equal))`, where `num_labels
    = CIN + CIP + CRN + CRP + UIN + UIP + URN + URP`, `frac_covered = (CIN + CIP + CRN + CRP) / num_labels` and
    `frac_equal = (CIN + CRP + UIN + URP) / num_labels`. The division by zero evaluates to 1, by definition.
    """

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 num_covered_equal = cin + crp
        cdef float64 num_uncovered_equal = uin + urp
        cdef float64 num_equal = num_uncovered_equal + num_covered_equal
        cdef float64 num_covered = num_covered_equal + cip + crn
        cdef float64 num_uncovered = num_uncovered_equal + uip + urn
        cdef float64 num_labels = num_covered + num_uncovered

        if num_labels == 0:
            return 1

        cdef float64 diff = (num_covered / num_labels) - (num_equal / num_labels)

        if diff == 0:
            return 1

        return num_uncovered / (num_labels * diff)


cdef class FMeasure(Heuristic):
    """
    A heuristic that calculates as the (weighted) harmonic mean between the heuristics `Precision` and `Recall`, where
    the parameter `beta` allows to trade-off between both heuristics. If `beta == 1`, both heuristics are weighed
    equally. As `beta` approaches zero, the heuristics becomes equivalent to `Precision`. As `beta` approaches infinity,
    the heuristic becomes equivalent to `Recall`.
    """

    def __cinit__(self, float64 beta):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.beta = beta
        self.recall = Recall()
        self.precision = Precision()

    cdef float64 evaluate_confusion_matrix(self, float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                           float64 uip, float64 urn, float64 urp):
        cdef float64 beta = self.beta
        cdef Heuristic precision, recall
        cdef float64 r, p, beta_pow, denominator

        if isinf(beta):
            # Equivalent to recall
            recall = self.recall
            return recall.evaluate_confusion_matrix(cin, cip, crn, crp, uin, uip, urn, urp)
        elif beta > 0:
            # Weighted harmonic mean between recall and precision
            recall = self.recall
            precision = self.precision
            r = recall.evaluate_confusion_matrix(cin, cip, crn, crp, uin, uip, urn, urp)
            p = precision.evaluate_confusion_matrix(cin, cip, crn, crp, uin, uip, urn, urp)
            beta_pow = pow(beta, 2)
            denominator = beta_pow * p + r

            if denominator == 0:
                return 1

            return ((1 + beta_pow) * p * r) / denominator
        else:
            # Equivalent to precision
            precision = self.precision
            return precision.evaluate_confusion_matrix(cin, cip, crn, crp, uin, uip, urn, urp)
