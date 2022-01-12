/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/rule_evaluation/lift_function.hpp"


namespace seco {

    /**
     * Allows to configure a lift function that monotonously increases until a certain number of labels, where the
     * maximum lift is reached, and monotonously decreases afterwards.
     */
    class PeakLiftFunctionConfig final : public ILiftFunctionConfig {

        private:

            // TODO Move to the PeakLiftFunctionFactory
            uint32 numLabels_;

            uint32 peakLabel_;

            float64 maxLift_;

            float64 curvature_;

        public:

            PeakLiftFunctionConfig();

            /**
             * Returns the total number of available labels.
             *
             * @return The total number of available labels
             */
            uint32 getNumLabels() const;

            /**
             * Sets the total number of available labels.
             *
             * @param numLabels The total number of available labels. Must be greater than 0
             * @return          A reference to an object of type `PeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            PeakLiftFunctionConfig& setNumLabels(uint32 numLabels);

            /**
             * Returns the index of the label for which the lift is maximal.
             *
             * @return The index of the label for which the lift is maximal
             */
            uint32 getPeakLabel() const;

            /**
             * Sets the index of the label for which the lift is maximal.
             *
             * @param peakLabel The index of the label for which the lift is maximal.  Must be at least 0
             * @return          A reference to an object of type `PeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            // TODO Allow to use choose the peak label automatically based on the label cardinality
            PeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel);

            /**
             * Returns the lift at the peak label.
             *
             * @return The lift at the peak label
             */
            float64 getMaxLift() const;

            /**
             * Sets the lift at the peak label.
             *
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @return          A reference to an object of type `PeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            PeakLiftFunctionConfig& setMaxLift(float64 maxLift);

            /**
             * Returns the curvature of the lift function.
             *
             * @return The curvature of the lift function
             */
            float64 getCurvature() const;

            /**
             * Sets the curvature of the lift function.
             *
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             * @return          A reference to an object of type `PeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            PeakLiftFunctionConfig& setCurvature(float64 curvature);

    };

    /**
     * Allows to create instances of the type `ILiftFunction` that monotonously increase until a certain number of
     * labels, where the maximum lift is reached, and monotonously decrease afterwards.
     */
    class PeakLiftFunctionFactory final : public ILiftFunctionFactory {

        private:

            uint32 numLabels_;

            uint32 peakLabel_;

            float64 maxLift_;

            float64 curvature_;

        public:

            /**
             * @param numLabels The total number of available labels. Must be greater than 0
             * @param peakLabel The index of the label for which the lift is maximal. Must be at least 0
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             */
            PeakLiftFunctionFactory(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature);

            std::unique_ptr<ILiftFunction> create() const override;

    };

}
