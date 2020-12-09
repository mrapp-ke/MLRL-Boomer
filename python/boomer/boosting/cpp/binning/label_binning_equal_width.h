/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "label_binning.h"


namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients, in a way such that each bin contains labels with
     * gradients from equally sized value ranges.
     *
     * @tparam T The type of the vector that provides access to the gradients
     */
    template<class T>
    class EqualWidthLabelBinning : public ILabelBinning<T> {

        public:

            LabelInfo getLabelInfo(T& statisticVector, uint32 numPositiveBins, uint32 numNegativeBins) const override;

            void createBins(uint32 numPositiveBins, uint32 numNegativeBins, const T& statisticVector,
                            IBinningObserver<float64>& observer) const override;

    };

}
