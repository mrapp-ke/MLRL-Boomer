/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

/**
 * Defines an interface for all classes that provide access to the statistics space.
 */
class IStatisticsSpace {
    public:

        virtual ~IStatisticsSpace() {}

        /**
         * Returns the number of statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of outputs.
         *
         * @return The number of outputs
         */
        virtual uint32 getNumOutputs() const = 0;
};

/**
 * An abstract base class for all classes that provide access to the statistics space.
 *
 * @tparam State The type of the state of the training process
 */
template<typename State>
class AbstractStatisticsSpace : virtual public IStatisticsSpace {
    protected:

        /**
         * A reference to an object of template type `State` that represents the state of the training process.
         */
        State& state_;

    public:

        /**
         * @param state A reference to an object of template type `State` that represents the state of the training
         *              process
         */
        AbstractStatisticsSpace(State& state) : state_(state) {}

        /**
         * @see `IStatisticsSpace::getNumStatistics`
         */
        uint32 getNumStatistics() const override final {
            return state_.statisticMatrixPtr->getNumRows();
        }

        /**
         * @see `IStatisticsSpace::getNumOutputs`
         */
        uint32 getNumOutputs() const override final {
            return state_.statisticMatrixPtr->getNumCols();
        }
};
