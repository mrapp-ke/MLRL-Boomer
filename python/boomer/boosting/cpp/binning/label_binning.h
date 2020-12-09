/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/binning/binning_observer.h"


namespace boosting {

    /**
     * Stores information about a vector that provides access to the statistics for individual labels. This includes the
     * number of positive and negative bins, the labels should be assigned to, as well as the minimum and maximum
     * statistic in the vector.
     */
    struct LabelInfo {
        uint32 numPositiveBins;
        float64 minPositive;
        float64 maxPositive;
        uint32 numNegativeBins;
        float64 minNegative;
        float64 maxNegative;
    };

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
             * Retrieves and returns information about the statistics for individual labels in a given vector that is
             * required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param statisticVector   A reference to an object of template type `T` that provides access to the
             *                          statistics for individual labels
             * @param numPositiveBins   The number of bins to be used for labels that should be predicted positively.
             *                          Must be at least 1
             * @param numNegativeBins   The number of bins to be used for labels that should be predicted negatively.
             *                          Must be at least 1
             * @return                  A struct of `type `LabelInfo` that stores the information
             */
            virtual LabelInfo getLabelInfo(const T& statisticVector, uint32 numPositiveBins,
                                           uint32 numNegativeBins) const = 0;

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
