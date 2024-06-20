/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/learner.hpp"
#include "mlrl/common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "mlrl/common/sampling/instance_sampling_stratified_output_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"

#include <memory>
#include <utility>

/**
 * Defines an interface for all rule learners that can be applied to classification problems.
 */
class MLRLCOMMON_API IClassificationRuleLearner : virtual public IRuleLearner {
    public:

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use label-wise stratified
         * instance sampling.
         */
        class IOutputWiseStratifiedInstanceSamplingMixin : virtual public IClassificationRuleLearner::IConfig {
            public:

                virtual ~IOutputWiseStratifiedInstanceSamplingMixin() override {}

                /**
                 * Configures the rule learner to sample from the available training examples using stratification, such
                 * that for each label the proportion of relevant and irrelevant examples is maintained, whenever a new
                 * rule should be learned.
                 *
                 * @return A reference to an object of type `IOutputWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual IOutputWiseStratifiedInstanceSamplingConfig& useOutputWiseStratifiedInstanceSampling() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<OutputWiseStratifiedInstanceSamplingConfig> ptr =
                      std::make_unique<OutputWiseStratifiedInstanceSamplingConfig>();
                    IOutputWiseStratifiedInstanceSamplingConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use example-wise stratified
         * instance sampling.
         */
        class IExampleWiseStratifiedInstanceSamplingMixin : virtual public IClassificationRuleLearner::IConfig {
            public:

                virtual ~IExampleWiseStratifiedInstanceSamplingMixin() override {}

                /**
                 * Configures the rule learner to sample from the available training examples using stratification,
                 * where distinct label vectors are treated as individual classes, whenever a new rule should be
                 * learned.
                 *
                 * @return A reference to an object of type `IExampleWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
                      std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
                    IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to partition the available
         * training examples into a training set and a holdout set using stratification, such that for each label the
         * proportion of relevant and irrelevant examples is maintained.
         */
        class IOutputWiseStratifiedBiPartitionSamplingMixin : virtual public IClassificationRuleLearner::IConfig {
            public:

                virtual ~IOutputWiseStratifiedBiPartitionSamplingMixin() override {}

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, such that for each label the proportion of relevant and irrelevant
                 * examples is maintained.
                 *
                 * @return A reference to an object of type `IOutputWiseStratifiedBiPartitionSamplingConfig` that allows
                 *         further configuration of the method for partitioning the available training examples into a
                 *         training and a holdout set
                 */
                virtual IOutputWiseStratifiedBiPartitionSamplingConfig& useOutputWiseStratifiedBiPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    std::unique_ptr<OutputWiseStratifiedBiPartitionSamplingConfig> ptr =
                      std::make_unique<OutputWiseStratifiedBiPartitionSamplingConfig>();
                    IOutputWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
                    partitionSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to partition the available
         * training examples into a training set and a holdout set using stratification, where distinct label vectors
         * are treated as individual classes.
         */
        class IExampleWiseStratifiedBiPartitionSamplingMixin : virtual public IClassificationRuleLearner::IConfig {
            public:

                virtual ~IExampleWiseStratifiedBiPartitionSamplingMixin() override {}

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, where distinct label vectors are treated as individual classes
                 *
                 * @return A reference to an object of type `IExampleWiseStratifiedBiPartitionSamplingConfig` that
                 *         allows further configuration of the method for partitioning the available training examples
                 *         into a training and a holdout set
                 */
                virtual IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
                      std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
                    IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
                    partitionSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        virtual ~IClassificationRuleLearner() override {}
};
