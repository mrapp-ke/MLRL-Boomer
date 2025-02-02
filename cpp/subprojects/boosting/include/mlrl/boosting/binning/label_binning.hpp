/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_sparse.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable.hpp"
#include "mlrl/boosting/util/blas.hpp"
#include "mlrl/boosting/util/lapack.hpp"

#include <functional>
#include <memory>

namespace boosting {

    /**
     * Stores information about a vector that provides access to the binning criteria for individual labels. This
     * includes the number of positive and negative bins, the labels should be assigned to, as well as the minimum and
     * maximum criterion in the vector.
     */
    template<typename CriterionType>
    struct LabelInfo final {
        public:

            /**
             * The number of positive bins.
             */
            uint32 numPositiveBins;

            /**
             * The minimum among all statistics that belong to the positive bins.
             */
            CriterionType minPositive;

            /**
             * The maximum among all statistics that belong to the positive bins.
             */
            CriterionType maxPositive;

            /**
             * The number of negative bins.
             */
            uint32 numNegativeBins;

            /**
             * The minimum among all statistics that belong to the negative bins.
             */
            CriterionType minNegative;

            /**
             * The maximum among all statistics that belong to the negative bins.
             */
            CriterionType maxNegative;
    };

    /**
     * Defines an interface for methods that assign labels to bins, based on the corresponding gradients and Hessians.
     */
    class ILabelBinning {
        public:

            virtual ~ILabelBinning() {}

            /**
             * A callback function that is invoked when a label is assigned to a bin. It takes the index of the bin and
             * the index of the label as arguments.
             */
            typedef std::function<void(uint32 binIndex, uint32 labelIndex)> Callback;

            /**
             * A callback function that is invoked when a label with zero statistics is encountered. It takes the index
             * of the label as an argument.
             */
            typedef std::function<void(uint32 labelIndex)> ZeroCallback;

            /**
             * Returns an upper bound for the number of bins used by the binning method, given a specific number of
             * labels for which rules may predict.
             *
             * @param numLabels The number of labels for which rules may predict
             * @return          The maximum number of bins used by the binning method
             */
            virtual uint32 getMaxBins(uint32 numLabels) const = 0;

            /**
             * Retrieves and returns information that is required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param criteria      An iterator to the label-wise criteria that should be used to assign individual
             *                      labels to bins
             * @param numCriteria   The number of label-wise criteria
             * @return              A struct of type `LabelInfo` that stores the information
             */
            virtual LabelInfo<float64> getLabelInfo(View<float64>::const_iterator criteria,
                                                    uint32 numCriteria) const = 0;

            /**
             * Assigns the labels to bins based on label-wise criteria.
             *
             * @param labelInfo     A struct of type `LabelInfo` that stores information that is required to apply the
             *                      binning method
             * @param criteria      An iterator to the label-wise criteria that should be used to assign individual
             *                      labels to bins
             * @param numCriteria   The number of label-wise criteria
             * @param callback      A callback that is invoked when a label is assigned to a bin
             * @param zeroCallback  A callback that is invoked when a label for which the criterion is zero is
             *                      encountered
             */
            virtual void createBins(LabelInfo<float64> labelInfo, View<float64>::const_iterator criteria,
                                    uint32 numCriteria, Callback callback, ZeroCallback zeroCallback) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelBinning`.
     */
    class ILabelBinningFactory {
        public:

            virtual ~ILabelBinningFactory() {}

            /**
             * Creates and returns a new object of type `ILabelBinning`.
             *
             * @return An unique pointer to an object of type `ILabelBinning` that has been created
             */
            virtual std::unique_ptr<ILabelBinning> create() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a method that assigns labels to bins.
     */
    class ILabelBinningConfig {
        public:

            virtual ~ILabelBinningConfig() {}

            /**
             * Creates and returns a new object of type `IDecomposableRuleEvaluationFactory` that allows to calculate
             * the predictions of complete rules according to the specified configuration.
             *
             * @return An unique pointer to an object of type `IDecomposableRuleEvaluationFactory` that has been created
             */
            virtual std::unique_ptr<IDecomposableRuleEvaluationFactory>
              createDecomposableCompleteRuleEvaluationFactory() const = 0;

            /**
             * Creates and returns a new object of type `ISparseDecomposableRuleEvaluationFactory` that allows to
             * calculate the prediction of partial rules, which predict for a predefined number of outputs, according to
             * the specified configuration.
             *
             * @param outputRatio   A percentage that specifies for how many outputs the rule heads should predict
             * @param minOutputs    The minimum number of outputs for which the rule heads should predict
             * @param maxOutputs    The maximum number of outputs for which the rule heads should predict
             * @return              An unique pointer to an object of type `ISparseDecomposableRuleEvaluationFactory`
             *                      that has been created
             */
            virtual std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                  uint32 maxOutputs) const = 0;

            /**
             * Creates and returns a new object of type `ISparseDecomposableRuleEvaluationFactory` that allows to
             * calculate the prediction of partial rules, which predict for a subset of the available outputs that is
             * determined dynamically, according to the specified configuration.
             *
             * @param threshold A threshold that affects for how many outputs the rule heads should predict
             * @param exponent  An exponent that is used to weigh the estimated predictive quality for individual
             *                  outputs
             * @return          An unique pointer to an object of type `ISparseDecomposableRuleEvaluationFactory` that
             *                  has been created
             */
            virtual std::unique_ptr<ISparseDecomposableRuleEvaluationFactory>
              createDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent) const = 0;

            /**
             * Creates and returns a new object of type `INonDecomposableRuleEvaluationFactory` that allows to calculate
             * the predictions of complete rules according to the specified configuration.
             *
             * @param blasFactory   A reference to an object of type `BlasFactory` that allows to create objects for
             *                      executing BLAS routines
             * @param lapackFactory A reference to an object of type `LapackFactory` that allows to create objects for
             *                      executing LAPACK routines
             * @return              An unique pointer to an object of type `INonDecomposableRuleEvaluationFactory` that
             *                      has been created
             */
            virtual std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableCompleteRuleEvaluationFactory(const BlasFactory& blasFactory,
                                                                 const LapackFactory& lapackFactory) const = 0;

            /**
             * Creates and returns a new object of type `INonDecomposableRuleEvaluationFactory` that allows to calculate
             * the predictions of partial rules, which predict for a predefined number of outputs, according to the
             * specified configuration.
             *
             * @param outputRatio   A percentage that specifies for how many outputs the rule heads should predict
             * @param minOutputs    The minimum number of outputs for which the rule heads should predict
             * @param maxOutputs    The maximum number of outputs for which the rule heads should predict
             * @param blasFactory   A reference to an object of type `BlasFactory` that allows to create objects for
             *                      executing BLAS routines
             * @param lapackFactory A reference to an object of type `LapackFactory` that allows to create objects for
             *                      executing LAPACK routines
             * @return              An unique pointer to an object of type `INonDecomposableRuleEvaluationFactory` that
             *                      has been created
             */
            virtual std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableFixedPartialRuleEvaluationFactory(float32 outputRatio, uint32 minOutputs,
                                                                     uint32 maxOutputs, const BlasFactory& blasFactory,
                                                                     const LapackFactory& lapackFactory) const = 0;

            /**
             * Creates and returns a new object of type `INonDecomposableRuleEvaluationFactory` that allows to calculate
             * the predictions of partial rules, which predict for a subset of the available labels that is determined
             * dynamically, according to the specified configuration.
             *
             * @param threshold     A threshold that affects for how many labels the rule heads should predict
             * @param exponent      An exponent that is used to weigh the estimated predictive quality for individual
             *                      labels
             * @param blasFactory   A reference to an object of type `BlasFactory` that allows to create objects for
             *                      executing BLAS routines
             * @param lapackFactory A reference to an object of type `LapackFactory` that allows to create objects for
             *                      executing LAPACK routines
             * @return              An unique pointer to an object of type `INonDecomposableRuleEvaluationFactory` that
             *                      has been created
             */
            virtual std::unique_ptr<INonDecomposableRuleEvaluationFactory>
              createNonDecomposableDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent,
                                                                       const BlasFactory& blasFactory,
                                                                       const LapackFactory& lapackFactory) const = 0;
    };

}
