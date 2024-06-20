"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.common.cython.instance_sampling import ExampleWiseStratifiedInstanceSamplingConfig, \
    OutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.partition_sampling import ExampleWiseStratifiedBiPartitionSamplingConfig, \
    OutputWiseStratifiedBiPartitionSamplingConfig


class OutputWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use label-wise stratified instance sampling.
    """

    @abstractmethod
    def use_output_wise_stratified_instance_sampling(self) -> OutputWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, such that for
        each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should be
        learned.

        :return: An `OutputWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class ExampleWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use example-wise stratified instance sampling.
    """

    @abstractmethod
    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, where distinct
        label vectors are treated as individual classes, whenever a new rule should be learned.

        :return: An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class OutputWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    @abstractmethod
    def use_output_wise_stratified_bi_partition_sampling(self) -> OutputWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.

        :return: An `OutputWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass


class ExampleWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, where distinct label vectors are treated as individual classes.
    """

    @abstractmethod
    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, where distinct label vectors are treated as individual classes

        :return: An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass
