/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/dll_exports.hpp"

#include <memory>

// Forward declarations
class IInstanceSampling;
class IClassificationInstanceSamplingFactory;
class IPartition;
class IRowWiseLabelMatrix;
class IRowWiseRegressionMatrix;
class IRegressionInstanceSamplingFactory;
class IStatistics;

/**
 * Defines an interface for all classes that provide access to the weights of individual training examples.
 */
class MLRLCOMMON_API IExampleWeights {
    public:

        virtual ~IExampleWeights() {}

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in classification
         * problems, based on the type of this partition.
         *
         * @param factory       A reference to an object of type `IClassificationInstanceSamplingFactory` that should be
         *                      used to create the instance
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of individual training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @param partition     A reference to an object of type `IPartition` that provides access to the indices of
         *                      the training examples that are included in the training and holdout set, respectively
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IClassificationInstanceSamplingFactory& factory, const IRowWiseLabelMatrix& labelMatrix,
          IStatistics& statistics, IPartition& partition) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling` that can be used in classification
         * problems, based on the type of this partition.
         *
         * @param factory           A reference to an object of type `IRegressionInstanceSamplingFactory` that should be
         *                          used to create the instance
         * @param regressionMatrix  A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
         *                          access to the regression scores of individual training examples
         * @param statistics        A reference to an object of type `IStatistics` that provides access to the
         *                          statistics which serve as a basis for learning rules
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that are included in the training and holdout set,
         *                          respectively
         * @return                  An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(
          const IRegressionInstanceSamplingFactory& factory, const IRowWiseRegressionMatrix& regressionMatrix,
          IStatistics& statistics, IPartition& partition) const = 0;
};
