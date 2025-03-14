/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/seco/lift_functions/lift_function.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

#include <memory>

namespace seco {

    /**
     * Defines an interface for all classes that allow to configure a lift function that monotonously increases until a
     * certain number of labels, where the maximum lift is reached, and monotonously decreases afterwards.
     */
    class MLRLSECO_API IPeakLiftFunctionConfig {
        public:

            virtual ~IPeakLiftFunctionConfig() {}

            /**
             * Returns the number of labels for which the lift is maximal.
             *
             * @return The number of labels for which the lift is maximal or 0, if the average label cardinality is used
             */
            virtual uint32 getPeakLabel() const = 0;

            /**
             * Sets the number of labels for which the lift should be maximal.
             *
             * @param peakLabel The number of labels for which the lift should be maximal. Must be at least 1 or 0, if
             *                  the average label cardinality should be used
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) = 0;

            /**
             * Returns the lift at the peak label.
             *
             * @return The lift at the peak label
             */
            virtual float32 getMaxLift() const = 0;

            /**
             * Sets the lift at the peak label.
             *
             * @param maxLift   The lift at the peak label. Must be at least 1
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setMaxLift(float32 maxLift) = 0;

            /**
             * Returns the curvature of the lift function.
             *
             * @return The curvature of the lift function
             */
            virtual float32 getCurvature() const = 0;

            /**
             * Sets the curvature of the lift function.
             *
             * @param curvature The curvature of the lift function. A greater value results in a steeper curvature, a
             *                  smaller value results in a flatter curvature. Must be greater than 0
             * @return          A reference to an object of type `IPeakLiftFunctionConfig` that allows further
             *                  configuration of the lift function
             */
            virtual IPeakLiftFunctionConfig& setCurvature(float32 curvature) = 0;
    };

    /**
     * Allows to configure a lift function that monotonously increases until a certain number of labels, where the
     * maximum lift is reached, and monotonously decreases afterwards.
     */
    class PeakLiftFunctionConfig final : public ILiftFunctionConfig,
                                         public IPeakLiftFunctionConfig {
        private:

            uint32 peakLabel_;

            float32 maxLift_;

            float32 curvature_;

        public:

            PeakLiftFunctionConfig();

            uint32 getPeakLabel() const override;

            IPeakLiftFunctionConfig& setPeakLabel(uint32 peakLabel) override;

            float32 getMaxLift() const override;

            IPeakLiftFunctionConfig& setMaxLift(float32 maxLift) override;

            float32 getCurvature() const override;

            IPeakLiftFunctionConfig& setCurvature(float32 curvature) override;

            std::unique_ptr<ILiftFunctionFactory> createLiftFunctionFactory(
              const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
