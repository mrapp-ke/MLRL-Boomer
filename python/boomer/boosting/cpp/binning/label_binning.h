/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/binning/binning_observer.h"


namespace boosting {

    /**
     * Defines an interface for methods that assign labels to bins, based on the corresponding gradients.
     *
     * @tparam T The type of the vector that provides access to the gradients
     */
    template<class T>
    class ILabelBinning {

        public:

            virtual ~ILabelBinning() { };

            /**
             * Assigns the labels to bins, based on the corresponding gradients.
             *
             * @param numPositiveBins   The number of bins to be used for labels that should be predicted positively.
             *                          Must be at least 1
             * @param numNegativeBins   The number of bins to be used for labels that should be predicted negatively.
             *                          Must be at least 1
             * @param statisticVector   A reference to an object of template type `T` that provides access to the
             *                          gradients
             * @param observer          A reference to an object of type `IBinningObserver` that should be notified when
             *                          a label is assigned to a bin
             */
            virtual void createBins(uint32 numPositiveBins, uint32 numNegativeBins, const T& statisticVector,
                                    IBinningObserver<float64>& observer) const = 0;

    };

}
