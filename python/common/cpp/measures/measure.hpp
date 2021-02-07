/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/view_c_contiguous.hpp"
#include "../input/label_vector.hpp"


/**
 * Defines an interface for all measures that may be used to assess the quality of predictions for certain examples by
 * comparing them to the corresponding ground truth labels.
 */
class IMeasure {

    public:

        virtual ~IMeasure() { };

        /**
         * Calculates and returns a numerical score that assess the quality of predictions for the example at a specific
         * index by comparing them to the corresponding ground truth labels.
         *
         * @param exampleIndex  The index of the example
         * @param labelVector   A reference to an object of type `LabelVector` that provides access to the relevant
         *                      labels of the given example
         * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the predicted scores
         * @return              The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const LabelVector& labelVector,
                                 const CContiguousView<float64>& scoreMatrix) const = 0;

};
