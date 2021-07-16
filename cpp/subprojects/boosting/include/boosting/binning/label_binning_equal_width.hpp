/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients and Hessians, in a way such that each bin contains
     * labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinning final : public ILabelBinning {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used to assign labels to, e.g., if
             *                  100 labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
             * @param minBins   The minimum number of bins to be used to assign labels to. Must be at least 2
             * @param maxBins   The maximum number of bins to be used to assign labels to. Must be at least `minBins` or
             *                  0, if the maximum number of bins should not be restricted
             */
            EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins);

            uint32 getMaxBins(uint32 numLabels) const override;

            LabelInfo getLabelInfo(DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                   DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                   DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                                   DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                                   float64 l2RegularizationWeight) const override;

            LabelInfo getLabelInfo(DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                                   DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                                   DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                                   DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                                   float64 l2RegularizationWeight) const override;

            void createBins(LabelInfo labelInfo, DenseLabelWiseStatisticVector::gradient_const_iterator gradientsBegin,
                            DenseLabelWiseStatisticVector::gradient_const_iterator gradientsEnd,
                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansBegin,
                            DenseLabelWiseStatisticVector::hessian_const_iterator hessiansEnd,
                            float64 l2RegularizationWeight, Callback callback,
                            ZeroCallback zeroCallback) const override;

            void createBins(LabelInfo labelInfo,
                            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsBegin,
                            DenseExampleWiseStatisticVector::gradient_const_iterator gradientsEnd,
                            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansBegin,
                            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessiansEnd,
                            float64 l2RegularizationWeight, Callback callback,
                            ZeroCallback zeroCallback) const override;

    };

}
