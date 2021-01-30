/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "statistics.h"
#include "../input/label_matrix.h"


/**
 * Provides access to an object of type `IStatistics`.
 */
class IStatisticsProvider {

    public:

        virtual ~IStatisticsProvider() { };

        /**
         * Returns an object of type `IStatistics`.
         *
         * @return A reference to an object of type `IStatistics`
         */
        virtual IStatistics& get() const = 0;

        /**
         * Allows to switch the implementation that is used for calculating the predictions of rules, as well as
         * corresponding quality scores, from the one that was initially used for learning the default rule, to another
         * one that will be used for all remaining rules.
         */
        virtual void switchRuleEvaluation() = 0;

};

/**
 * Defines an interface for all classes that allow to create instances of the class `IStatisticsProvider`.
 */
class IStatisticsProviderFactory {

    public:

        virtual ~IStatisticsProviderFactory() { };

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`.
         *
         * @param labelMatrixPtr    A shared pointer to an object of type `IRandomAccessLabelMatrix` that provides
         *                          access to the labels of the training examples
         * @return                  An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> create(
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr) const = 0;

};
