"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.utility cimport move

from _io import StringIO


cdef class RuleModel:
    """
    A wrapper for the C++ class `RuleModel`.
    """
    pass


cdef class ModelBuilder:
    """
    A wrapper for the pure virtual C++ class `IModelBuilder`.
    """

    cdef RuleModel build(self):
        """
        Builds and returns the model.

        :return: The model that has been built
        """
        cdef RuleModel model = RuleModel()
        model.model_ptr = move(self.model_builder_ptr.get().build())
        return model


cdef uint32 __format_conditions(uint32 num_processed_conditions, uint32 num_conditions,
                                ConjunctiveBodyImpl.index_const_iterator index_iterator,
                                ConjunctiveBodyImpl.threshold_const_iterator threshold_iterator, object attributes,
                                bint print_feature_names, bint print_nominal_values, object text, object comparator):
    cdef uint32 result = num_processed_conditions
    cdef uint32 feature_index, i
    cdef object attribute

    for i in range(num_conditions):
        if result > 0:
            text.write(' & ')

        feature_index = index_iterator[i]
        attribute = attributes[feature_index] if len(attributes) > feature_index else None

        if print_feature_names and attribute is not None:
            text.write(attribute.attribute_name)
        else:
            text.write(str(feature_index))

        text.write(' ')
        text.write(comparator)
        text.write(' ')

        if print_nominal_values and attribute is not None and attribute.nominal_values is not None and len(
                attribute.nominal_values) > i:
            text.write('"' + attribute.nominal_values[i] + '"')
        else:
            text.write(str(threshold_iterator[i]))

        result += 1

    return result


cdef class RuleModelFormatter:
    """
    Allows to create textual representations of the rules that are contained by a `RuleModel`.
    """

    def __cinit__(self, list attributes, list label_names, bint print_feature_names=True, bint print_label_names=True,
                  bint print_nominal_values=False):
        """
        :param attributes:              A list that contains the attributes
        :param label_names:             A list that contains the names of the labels
        :param print_feature_names:     True, if the names of features should be printed, False otherwise
        :param print_label_names:       True, if the names of labels should be printed, False otherwise
        :param print_nominal_values:    True, if the values of nominal values should be printed, False otherwise
        """
        self.print_feature_names = print_feature_names
        self.print_label_names = print_label_names
        self.print_nominal_values = print_nominal_values
        self.attributes = attributes
        self.label_names = label_names
        self.text = StringIO()

    cdef __visit_empty_body(self, const EmptyBodyImpl& body):
        self.text.write('{}')

    cdef __visit_conjunctive_body(self, const ConjunctiveBodyImpl& body):
        cdef object text = self.text
        cdef bint print_feature_names = self.print_feature_names
        cdef bint print_nominal_values = self.print_nominal_values
        cdef list attributes = self.attributes
        cdef uint32 num_processed_conditions = 0

        text.write('{')

        cdef ConjunctiveBodyImpl.threshold_const_iterator threshold_iterator = body.leq_thresholds_cbegin()
        cdef ConjunctiveBodyImpl.index_const_iterator index_iterator = body.leq_indices_cbegin()
        cdef uint32 num_conditions = body.getNumLeq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '<=')

        threshold_iterator = body.gr_thresholds_cbegin()
        index_iterator = body.gr_indices_cbegin()
        num_conditions = body.getNumGr()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '>')

        threshold_iterator = body.eq_thresholds_cbegin()
        index_iterator = body.eq_indices_cbegin()
        num_conditions = body.getNumEq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '==')

        threshold_iterator = body.neq_thresholds_cbegin()
        index_iterator = body.neq_indices_cbegin()
        num_conditions = body.getNumNeq()
        num_processed_conditions = __format_conditions(num_processed_conditions, num_conditions, index_iterator,
                                                       threshold_iterator, attributes, print_feature_names,
                                                       print_nominal_values, text, '!=')

        text.write('}')

    cdef __visit_full_head(self, const FullHeadImpl& head):
        cdef object text = self.text
        cdef bint print_label_names = self.print_label_names
        cdef list label_names = self.label_names
        cdef FullHeadImpl.score_const_iterator score_iterator = head.scores_cbegin()
        cdef uint32 num_elements = head.getNumElements()
        cdef uint32 i

        text.write(' => (')

        for i in range(num_elements):
            if i > 0:
                text.write(', ')

            if print_label_names and len(label_names) > i:
                text.write(label_names[i])
            else:
                text.write(str(i))

            text.write(' = ')
            text.write('{0:.2f}'.format(score_iterator[i]))

        text.write(')\n')

    cdef __visit_partial_head(self, const PartialHeadImpl& head):
        cdef object text = self.text
        cdef bint print_label_names = self.print_label_names
        cdef list label_names = self.label_names
        cdef PartialHeadImpl.score_const_iterator score_iterator = head.scores_cbegin()
        cdef PartialHeadImpl.index_const_iterator index_iterator = head.indices_cbegin()
        cdef uint32 num_elements = head.getNumElements()
        cdef uint32 label_index, i

        text.write(' => (')

        for i in range(num_elements):
            if i > 0:
                text.write(', ')

            label_index = index_iterator[i]

            if print_label_names and len(label_names) > label_index:
                text.write(label_names[label_index])
            else:
                text.write(str(label_index))

            text.write(' = ')
            text.write('{0:.2f}'.format(score_iterator[i]))

        text.write(')\n')

    cpdef void format(self, RuleModel model):
        """
        Creates a textual representation of a specific model.

        :param model: The `RuleModel` to be formatted
        """
        model.model_ptr.get().visit(
            wrapEmptyBodyVisitor(<void*>self, <EmptyBodyCythonVisitor>self.__visit_empty_body),
            wrapConjunctiveBodyVisitor(<void*>self, <ConjunctiveBodyCythonVisitor>self.__visit_conjunctive_body),
            wrapFullHeadVisitor(<void*>self, <FullHeadCythonVisitor>self.__visit_full_head),
            wrapPartialHeadVisitor(<void*>self, <PartialHeadCythonVisitor>self.__visit_partial_head))

    cpdef object get_text(self):
        """
        Returns the textual representation that has been created via the `format` method.

        :return: The textual representation
        """
        return self.text.getvalue()
