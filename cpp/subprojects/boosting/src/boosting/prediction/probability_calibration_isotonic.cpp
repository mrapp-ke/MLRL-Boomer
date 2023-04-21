#include "boosting/prediction/probability_calibration_isotonic.hpp"

/**
 * A model for the calibration of marginal probabilities based on isotonic regression.
 */
class IsotonicMarginalProbabilityCalibrationModel final : public IIsotonicMarginalProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override {
            // TODO Implement
            return marginalProbability;
        }
};

/**
 * An implementation of the type `IMarginalProbabilityCalibrator` that does fit a model for the calibration of marginal
 * probabilities based on isotonic regression.
 */
class IsotonicMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
    public:

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const BiPartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IMarginalProbabilityCalibrator>
  IsotonicMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibrator() const {
    return std::make_unique<IsotonicMarginalProbabilityCalibrator>();
}

std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel> createIsotonicMarginalProbabilityCalibrationModel() {
    return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
}
