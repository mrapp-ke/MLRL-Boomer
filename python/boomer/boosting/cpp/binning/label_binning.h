/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/data/types.h"
#include <functional>


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
             * A callback function that is invoked when a value is assigned to a bin. It takes the index of the bin, the
             * original index of the value, as well as the value itself, as arguments.
             */
            typedef std::function<void(uint32 binIndex, uint32 originalIndex, float64 value)> Callback;

            /**
             * Returns an upper bound for the number of bins used by the binning method, given a specific number of
             * labels for which rules may predict.
             *
             * @param numLabels The number of labels for which rules may predict
             * @return          The maximum number of bins used by the binning method
             */
            virtual uint32 getMaxBins(uint32 numLabels) const = 0;

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
             * @return                  A struct of `type `LabelInfo` that stores the information
             */
            virtual LabelInfo getLabelInfo(const T& statisticVector) const = 0;

            /**
             * Assigns the labels to bins, based on the corresponding gradients.
             *
             * @param labelInfo         A struct of type `LabelInfo` that stores information about the statistics in the
             *                          given vector
             * @param statisticVector   A reference to an object of template type `T` that provides access to the
             *                          gradients
             * @param callback          A callback that is invoked when a value is assigned to a bin
             */
            virtual void createBins(LabelInfo labelInfo, const T& statisticVector, Callback callback) const = 0;

    };

}
