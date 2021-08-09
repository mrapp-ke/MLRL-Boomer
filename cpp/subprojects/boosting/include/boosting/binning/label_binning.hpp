/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include <functional>
#include <memory>


namespace boosting {

    /**
     * Stores information about a vector that provides access to the statistics for individual labels. This includes the
     * number of positive and negative bins, the labels should be assigned to, as well as the minimum and maximum
     * statistic in the vector.
     */
    struct LabelInfo {

        LabelInfo() : numPositiveBins(0), numNegativeBins(0) { };

        /**
         * The number of positive bins.
         */
        uint32 numPositiveBins;

        /**
         * The minimum among all statistics that belong to the positive bins.
         */
        float64 minPositive;

        /**
         * The maximum among all statistics that belong to the positive bins.
         */
        float64 maxPositive;

        /**
         * The number of negative bins.
         */
        uint32 numNegativeBins;

        /**
         * The minimum among all statistics that belong to the negative bins.
         */
        float64 minNegative;

        /**
         * The maximum among all statistics that belong to the negative bins.
         */
        float64 maxNegative;

    };

    /**
     * Defines an interface for methods that assign labels to bins, based on the corresponding gradients and Hessians.
     */
    class ILabelBinning {

        public:

            virtual ~ILabelBinning() { };

            /**
             * A callback function that is invoked when a label is assigned to a bin. It takes the index of the bin, the
             * index of the label, as well as the corresponding gradient and Hessians, as arguments.
             */
            // TODO Remove arguments "gradient" and "hessian"
            typedef std::function<void(uint32 binIndex, uint32 labelIndex, float64 gradient, float64 hessian)> Callback;

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
             * Retrieves and returns information about the statistics for individual labels in a given
             * `DenseLabelWiseStatisticVector` that is required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param gradientsBegin            A `DenseLabelWiseStatisticVector::gradient_const_iterator` to the
             *                                  beginning of the gradients
             * @param gradientsEnd              A `DenseLabelWiseStatisticVector::gradient_const_iterator` to the end of
             *                                  the gradients
             * @param hessiansBegin             A `DenseLabelWiseStatisticVector::hessian_const_iterator` to the
             *                                  beginning of the Hessians
             * @param hessiansEnd               A `DenseLabelWiseStatisticVector::hessian_const_iterator` to the end of
             *                                  the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @return                          A struct of `type `LabelInfo` that stores the information
             */
            // TODO Remove
            virtual LabelInfo getLabelInfo(DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                           DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                           DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                                           DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                                           float64 l2RegularizationWeight) const = 0;

            /**
             * Retrieves and returns information about the statistics for individual labels in a given
             * `DenseLabelWiseStatisticVector` that is required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param statisticVector           A reference to an object of type `DenseLabelWiseStatisticVector` that
             *                                  provides access to the statistics
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @return                          A struct of `type `LabelInfo` that stores the information
             */
            virtual LabelInfo getLabelInfo(const DenseLabelWiseStatisticVector& statisticVector,
                                           float64 l2RegularizationWeight) const = 0;

            /**
             * Retrieves and returns information about the statistics for individual labels in a given
             * `DenseExampleWiseStatisticVector` that is required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param gradientsBegin            A `DenseExampleWiseStatisticVector::gradient_const_iterator` to the
             *                                  beginning of the gradients
             * @param gradientsEnd              A `DenseExampleWiseStatisticVector::gradient_const_iterator` to the end
             *                                  of the gradients
             * @param hessiansBegin             A `DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator` to
             *                                  the beginning of the Hessians
             * @param hessiansEnd               A `DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator` to
             *                                  the end of the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @return                          A struct of `type `LabelInfo` that stores the information
             */
            // TODO Replace iterators with reference to object
            virtual LabelInfo getLabelInfo(
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                float64 l2RegularizationWeight) const = 0;

            /**
             * Assigns the labels to bins, based on the corresponding statistics in a `DenseLabelWiseStatisticVector`.
             *
             * @param labelInfo                 A struct of type `LabelInfo` that stores information about the
             *                                  statistics in the given vector
             * @param gradientsBegin            A `DenseLabelWiseStatisticVector::gradient_const_iterator` to the
             *                                  beginning of the gradients
             * @param gradientsEnd              A `DenseLabelWiseStatisticVector::gradient_const_iterator` to the end of
             *                                  the gradients
             * @param hessiansBegin             A `DenseLabelWiseStatisticVector::hessian_const_iterator` to the
             *                                  beginning of the Hessians
             * @param hessiansEnd               A `DenseLabelWiseStatisticVector::hessian_const_iterator` to the end of
             *                                  the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @param callback                  A callback that is invoked when a label is assigned to a bin
             * @param zeroCallback              A callback that is invoked when a label with zero statistics is
             *                                  encountered
             */
            // TODO Remove
            virtual void createBins(LabelInfo labelInfo,
                                    DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                    DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                    DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                                    DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                                    float64 l2RegularizationWeight, Callback callback,
                                    ZeroCallback zeroCallback) const = 0;

            /**
             * Assigns the labels to bins, based on the corresponding statistics in a `DenseExampleWiseStatisticVector`.
             *
             * @param labelInfo                 A struct of type `LabelInfo` that stores information about the
             *                                  statistics in the given vector
             * @param gradientsBegin            A `DenseExampleWiseStatisticVector::gradient_const_iterator` to the
             *                                  beginning of the gradients
             * @param gradientsEnd              A `DenseExampleWiseStatisticVector::gradient_const_iterator` to the end
             *                                  of the gradients
             * @param hessiansBegin             A `DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator` to
             *                                  the beginning of the Hessians
             * @param hessiansEnd               A `DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator` to
             *                                  the end of the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @param callback                  A callback that is invoked when a label is assigned to a bin
             * @param zeroCallback              A callback that is invoked when a label with zero statistics is
             *                                  encountered
             */
            // TODO Replace iterators with reference to object
            virtual void createBins(LabelInfo labelInfo,
                                    DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                    DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                                    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                                    float64 l2RegularizationWeight, Callback callback,
                                    ZeroCallback zeroCallback) const = 0;

    };

    /**
     * Defines an interface for all factories that allows to create instances of the type `ILabelBinning`.
     */
    class ILabelBinningFactory {

        public:

            virtual ~ILabelBinningFactory() { };

            /**
             * Creates and returns a new object of type `ILabelBinning`.
             *
             * @return An unique pointer to an object of type `ILabelBinning` that has been created
             */
            virtual std::unique_ptr<ILabelBinning> create() const = 0;

    };

}
