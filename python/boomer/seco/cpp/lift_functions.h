/**
 * Implements different lift functions that allow to affect the quality score of rules depending on how many labels they
 * predict.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"


namespace seco {

    /**
     * An abstract base class for all lift functions.
     */
    class AbstractLiftFunction {

        public:

            virtual ~AbstractLiftFunction();

            /**
             * Calculates and returns the lift for a specific number of labels.
             *
             * @param numLabels The number of labels for which the lift should be calculated
             * @return          The lift that has been calculated
             */
            virtual float64 calculateLift(intp numLabels);

            /**
             * Returns the maximum lift possible.
             *
             * @return The maximum lift possible
             */
            virtual float64 getMaxLift();

    };

    /**
     * A lift function that monotonously increases until a certain number of labels, where the maximum lift is reached,
     * and monotonously decreases afterwards.
     */
    class PeakLiftFunctionImpl : public AbstractLiftFunction {

        private:

            intp numLabels_;

            intp peakLabel_;

            float64 maxLift_;

            float64 curvature_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param peakLabel The number of labels for which the lift is maximum. Must be in [1, numLabels]
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             */
            PeakLiftFunctionImpl(intp numLabels, intp peakLabel, float64 maxLift, float64 curvature);

            ~PeakLiftFunctionImpl();

            float64 calculateLift(intp numLabels) override;

            float64 getMaxLift() override;

    };

}
