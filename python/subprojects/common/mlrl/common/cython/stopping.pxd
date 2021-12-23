from mlrl.common.cython._types cimport uint32, float64

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/stopping/stopping_criterion.hpp" nogil:

    cdef cppclass IStoppingCriterionFactory:
        pass


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass SizeStoppingCriterionFactoryImpl"SizeStoppingCriterionFactory"(IStoppingCriterionFactory):

        # Constructors:

        SizeStoppingCriterionFactoryImpl(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass TimeStoppingCriterionFactoryImpl"TimeStoppingCriterionFactory"(IStoppingCriterionFactory):

        # Constructors:

        TimeStoppingCriterionFactoryImpl(uint32 timeLimit) except +


cdef extern from "common/stopping/stopping_criterion_measure.hpp" nogil:

    cdef cppclass IAggregationFunctionFactory:
        pass


    cdef cppclass MinAggregationFunctionFactoryImpl"MinAggregationFunctionFactory"(IAggregationFunctionFactory):
        pass


    cdef cppclass MaxAggregationFunctionFactoryImpl"MaxAggregationFunctionFactory"(IAggregationFunctionFactory):
        pass


    cdef cppclass ArithmeticMeanAggregationFunctionFactoryImpl"ArithmeticMeanAggregationFunctionFactory"(
            IAggregationFunctionFactory):
        pass


    cdef cppclass MeasureStoppingCriterionFactoryImpl"MeasureStoppingCriterionFactory"(IStoppingCriterionFactory):

        # Constructors:

        MeasureStoppingCriterionFactoryImpl(unique_ptr[IAggregationFunctionFactory] aggregationFunctionFactoryPtr,
                                            uint32 minRules, uint32 updateInterval, uint32 stopInterval, uint32 numPast,
                                            uint32 numRecent, float64 minImprovement, bool forceStop) except +


cdef class StoppingCriterionFactory:

    # Attributes:

    cdef unique_ptr[IStoppingCriterionFactory] stopping_criterion_factory_ptr


cdef class SizeStoppingCriterionFactory(StoppingCriterionFactory):
    pass


cdef class TimeStoppingCriterionFactory(StoppingCriterionFactory):
    pass

cdef class AggregationFunctionFactory:

    # Attributes:

    cdef unique_ptr[IAggregationFunctionFactory] aggregation_function_factory_ptr


cdef class MinAggregationFunctionFactory(AggregationFunctionFactory):
    pass


cdef class MaxAggregationFunctionFactory(AggregationFunctionFactory):
    pass


cdef class ArithmeticMeanAggregationFunctionFactory(AggregationFunctionFactory):
    pass


cdef class MeasureStoppingCriterionFactory(StoppingCriterionFactory):
    pass
