/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "seco/lift_functions/lift_function.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a lift function that monotonously increases until a
     * certain number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
     */
    class IPeakLiftFunctionConfig {

        public:

            virtual ~IPeakLiftFunctionConfig() { };

            /**
             * Returns the number of labels for which the lift is maximal.
             *
             * @return The number of labels for which the lift is maximal
             */
            virtual uint32 getPeakLabel() const = 0;

            /**
             * Sets the number of labels for which the lift should be maximal.
             *
             * @param peakLabel The number of labels for which the lift should be maximal. Must be at least 1
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) = 0;

            /**
             * Returns the lift at the peak label.
             *
             * @return The lift at the peak label
             */
            virtual float64 getMaxLift() const = 0;

            /**
             * Sets the lift at the peak label.
             *
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setMaxLift(float64 maxLift) = 0;

            /**
             * Returns the curvature of the lift function.
             *
             * @return The curvature of the lift function
             */
            virtual float64 getCurvature() const = 0;

            /**
             * Sets the curvature of the lift function.
             *
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setCurvature(float64 curvature) = 0;

    };

    /**
     * Allows to configure a lift function that monotonously increases until a certain number of labels, where the
     * maximum lift is reached, and monotonously decreases afterwards.
     */
    class PeakLiftFunctionConfig final : public ILiftFunctionConfig, public IPeakLiftFunctionConfig {

        private:

            uint32 peakLabel_;

            float64 maxLift_;

            float64 curvature_;

        public:

            PeakLiftFunctionConfig();

            uint32 getPeakLabel() const override;

            // TODO Allow to use choose the peak label automatically based on the label cardinality
            IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) override;

            float64 getMaxLift() const override;

            IPeakLiftFunctionConfig& setMaxLift(float64 maxLift) override;

            float64 getCurvature() const override;

            IPeakLiftFunctionConfig& setCurvature(float64 curvature) override;

            std::unique_ptr<ILiftFunctionFactory> create(const ILabelMatrix& labelMatrix) const override;

    };

}
