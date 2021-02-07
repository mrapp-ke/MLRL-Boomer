/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "label_binning.hpp"


namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients, in a way such that each bin contains labels with
     * gradients from equally sized value ranges.
     *
     * @tparam T The type of the vector that provides access to the gradients
     */
    template<class T>
    class EqualWidthLabelBinning final : public ILabelBinning<T> {

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

            LabelInfo getLabelInfo(const T& statisticVector) const override;

            void createBins(LabelInfo labelInfo, const T& statisticVector, typename ILabelBinning<T>::Callback callback,
                            typename ILabelBinning<T>::ZeroCallback zeroCallback) const override;

    };

}
