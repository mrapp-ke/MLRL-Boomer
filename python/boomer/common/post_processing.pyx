"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to post-process rules once they have been learned.
"""


cdef class PostProcessor:
    """
    A base class for all classes that allow to post-process rules once they have been learned.
    """

    cdef void post_process(self, HeadCandidate* head):
        """
        Post-processes the head of a rule.

        :param head: A pointer to an object of type `HeadCandidate`, representing the head of the rule
        """
        pass
