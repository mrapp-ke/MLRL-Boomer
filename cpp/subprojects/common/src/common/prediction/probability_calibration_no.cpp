#include "common/prediction/probability_calibration_no.hpp"

class NoProbabilityCalibrationModel final : public INoProbabilityCalibrationModel {
    public:

        void calibrateProbabilities(VectorView<float64>::iterator probabilitiesBegin,
                                    VectorView<float64>::iterator probabilitiesEnd) const override {}
};

class NoProbabilityCalibrator final : public IProbabilityCalibrator {
    public:

        std::unique_ptr<IProbabilityCalibrationModel> fitCalibrationModel() const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }
};

class NoProbabilityCalibratorFactory final : public IProbabilityCalibratorFactory {
    public:

        std::unique_ptr<IProbabilityCalibrator> create() const override {
            return std::make_unique<NoProbabilityCalibrator>();
        }
};

std::unique_ptr<IProbabilityCalibratorFactory> NoProbabilityCalibratorConfig::createProbabilityCalibratorFactory() const {
    return std::make_unique<NoProbabilityCalibratorFactory>();
}

std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel() {
    return std::make_unique<NoProbabilityCalibrationModel>();
}
