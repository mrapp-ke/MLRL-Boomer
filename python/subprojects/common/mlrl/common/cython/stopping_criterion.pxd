from mlrl.common.cython._types cimport uint8, uint32, float64

from libcpp cimport bool


cdef extern from "common/stopping/stopping_criterion_size.hpp" nogil:

    cdef cppclass ISizeStoppingCriterionConfig:

        # Functions:

        uint32 getMaxRules() const

        ISizeStoppingCriterionConfig& setMaxRules(uint32 maxRules) except +


cdef extern from "common/stopping/stopping_criterion_time.hpp" nogil:

    cdef cppclass ITimeStoppingCriterionConfig:

        # Functions:

        uint32 getTimeLimit() const

        ITimeStoppingCriterionConfig& setTimeLimit(uint32 timeLimit) except +


cdef extern from "common/stopping/aggregation_function.hpp" nogil:

    cpdef enum AggregationFunctionImpl"AggregationFunction":

        MIN"AggregationFunction::MIN" = 0

        MAX"AggregationFunction::MAX" = 1

        ARITHMETIC_MEAN"AggregationFunction::ARITHMETIC_MEAN" = 2


cdef extern from "common/stopping/global_pre_pruning.hpp" nogil:

    cdef cppclass IPrePruningConfig:

        # Functions:

        AggregationFunctionImpl getAggregationFunction() const

        IPrePruningConfig& setAggregationFunction(AggregationFunctionImpl aggregationFunction) except +

        bool isHoldoutSetUsed() const

        IPrePruningConfig& setUseHoldoutSet(bool useHoldoutSet) except +

        uint32 getMinRules() const

        IPrePruningConfig& setMinRules(uint32 minRules) except +

        uint32 getUpdateInterval() const

        IPrePruningConfig& setUpdateInterval(uint32 updateInterval) except +

        uint32 getStopInterval() const;

        IPrePruningConfig& setStopInterval(uint32 stopInterval) except +

        uint32 getNumPast() const

        IPrePruningConfig& setNumPast(uint32 numPast) except +

        uint32 getNumCurrent() const

        IPrePruningConfig& setNumCurrent(uint32 numCurrent) except +

        float64 getMinImprovement() const

        IPrePruningConfig& setMinImprovement(float64 minImprovement) except +

        bool isStopForced() const

        IPrePruningConfig& setForceStop(bool forceStop) except +


cdef class SizeStoppingCriterionConfig:

    # Attributes:

    cdef ISizeStoppingCriterionConfig* config_ptr


cdef class TimeStoppingCriterionConfig:

    # Attributes:

    cdef ITimeStoppingCriterionConfig* config_ptr


cdef class EarlyStoppingCriterionConfig:

    # Attributes:

    cdef IPrePruningConfig* config_ptr
