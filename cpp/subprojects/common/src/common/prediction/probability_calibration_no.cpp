#include "common/prediction/probability_calibration_no.hpp"

/**
 * A model for the calibration of probabilities that does not make any adjustments.
 */
class NoProbabilityCalibrationModel final : public INoProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override {
            return marginalProbability;
        }

        float64 calibrateJointProbability(float64 jointProbability) const override {
            return jointProbability;
        }
};

/**
 * An implementation of the type `IProbabilityCalibrator` that does not fit a model for the calibration of
 * probabilities.
 */
class NoProbabilityCalibrator final : public IProbabilityCalibrator {
    public:

        std::unique_ptr<IProbabilityCalibrationModel> fitCalibrationModel() const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IProbabilityCalibrator> NoProbabilityCalibratorConfig::createProbabilityCalibrator() const {
    return std::make_unique<NoProbabilityCalibrator>();
}

std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel() {
    return std::make_unique<NoProbabilityCalibrationModel>();
}
