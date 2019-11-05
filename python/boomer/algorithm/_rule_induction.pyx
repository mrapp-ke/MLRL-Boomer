# cython: boundscheck=False
# cython: wraparound=False
from boomer.algorithm._model cimport float64, Rule, FullHead, EmptyBody
from boomer.algorithm._head_refinement cimport HeadRefinement
from boomer.algorithm._losses cimport Loss

cpdef Rule induce_default_rule(float64[::1, :] expected_scores, HeadRefinement head_refinement, Loss loss):
    """
    Induces the default rule that minimizes a certain loss function with respect to the expected confidence scores
    according to the ground truth.

    :param expected_scores: An array of dtype float, shape `(num_examples, num_labels)`, representing the expected
                            confidence scores according to the ground truth
    :param head_refinement: The 'HeadRefinement' to be used
    :param loss:            The loss function to be minimized
    :return:                The default rule that has been induced
    """
    cdef FullHead head = head_refinement.find_default_head(expected_scores, loss)
    cdef EmptyBody body = EmptyBody()
    cdef Rule rule = Rule(body, head)
    return rule
