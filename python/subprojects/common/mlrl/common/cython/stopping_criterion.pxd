from mlrl.common.cython._types cimport uint32, float64

from libcpp cimport bool


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass SizeStoppingCriterionConfigImpl"SizeStoppingCriterionConfig":

        # Functions:

        uint32 getMaxRules() const

        SizeStoppingCriterionConfigImpl& setMaxRules(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass TimeStoppingCriterionConfigImpl"TimeStoppingCriterionConfig":

        # Functions:

        uint32 getTimeLimit() const

        TimeStoppingCriterionConfigImpl& setTimeLimit(uint32 timeLimit) except +


cdef extern from "common/stopping/stopping_criterion_measure.hpp" nogil:

    cpdef enum AggregationFunctionImpl"MeasureStoppingCriterionConfig::AggregationFunction":

        MIN_"MeasureStoppingCriterionConfig::AggregationFunction::MIN",

        MAX_"MeasureStoppingCriterionConfig::AggregationFunction::MAX",

        ARITHMETIC_MEAN_"MeasureStoppingCriterionConfig::AggregationFunction::ARITHMETIC_MEAN"


    cdef cppclass MeasureStoppingCriterionConfigImpl"MeasureStoppingCriterionConfig":

        # Functions:

        AggregationFunctionImpl getAggregationFunction() const

        MeasureStoppingCriterionConfigImpl& setAggregationFunction(AggregationFunctionImpl aggregationFunction) except +

        uint32 getMinRules() const

        MeasureStoppingCriterionConfigImpl& setMinRules(uint32 minRules) except +

        uint32 getUpdateInterval() const

        MeasureStoppingCriterionConfigImpl& setUpdateInterval(uint32 updateInterval) except +

        uint32 getStopInterval() const;

        MeasureStoppingCriterionConfigImpl& setStopInterval(uint32 stopInterval) except +

        uint32 getNumPast() const

        MeasureStoppingCriterionConfigImpl& setNumPast(uint32 numPast) except +

        uint32 getNumCurrent() const

        MeasureStoppingCriterionConfigImpl& setNumCurrent(uint32 numCurrent) except +

        float64 getMinImprovement() const

        MeasureStoppingCriterionConfigImpl& setMinImprovement(float64 minImprovement) except +

        bool getForceStop() const

        MeasureStoppingCriterionConfigImpl& setForceStop(bool forceStop) except +


cdef class SizeStoppingCriterionConfig:

    # Attributes:

    cdef SizeStoppingCriterionConfigImpl* config_ptr


cdef class TimeStoppingCriterionConfig:

    # Attributes:

    cdef TimeStoppingCriterionConfigImpl* config_ptr


cdef enum AggregationFunction:

    MIN = <uint32>MIN_,

    MAX = <uint32>MAX_,

    ARITHMETIC_MEAN = <uint32>ARITHMETIC_MEAN_


cdef class MeasureStoppingCriterionConfig:


    # Attributes:

    cdef MeasureStoppingCriterionConfigImpl* config_ptr
