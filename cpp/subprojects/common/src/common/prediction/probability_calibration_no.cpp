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
 * An implementation of the type `IMarginalProbabilityCalibrator` that does not fit a model for the calibration of
 * probabilities.
 */
class NoProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
    public:

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IMarginalProbabilityCalibrator> NoProbabilityCalibratorConfig::createProbabilityCalibrator() const {
    return std::make_unique<NoProbabilityCalibrator>();
}

std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel() {
    return std::make_unique<NoProbabilityCalibrationModel>();
}
