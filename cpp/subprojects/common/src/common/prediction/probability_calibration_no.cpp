#include "common/prediction/probability_calibration_no.hpp"

/**
 * A model for the calibration of marginal probabilities that does not make any adjustments.
 */
class NoMarginalProbabilityCalibrationModel final : public INoMarginalProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override {
            return marginalProbability;
        }
};

/**
 * An implementation of the type `IMarginalProbabilityCalibrator` that does not fit a model for the calibration of
 * marginal probabilities.
 */
class NoMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
    public:

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
          const IStatistics& statistics) const override {
            return std::make_unique<NoMarginalProbabilityCalibrationModel>();
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
          BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
            return std::make_unique<NoMarginalProbabilityCalibrationModel>();
        }
};

std::unique_ptr<IMarginalProbabilityCalibrator> NoMarginalProbabilityCalibratorFactory::create() const {
    return std::make_unique<NoMarginalProbabilityCalibrator>();
}

bool NoMarginalProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
    return false;
}

std::unique_ptr<IMarginalProbabilityCalibratorFactory>
  NoMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibratorFactory() const {
    return std::make_unique<NoMarginalProbabilityCalibratorFactory>();
}

std::unique_ptr<INoMarginalProbabilityCalibrationModel> createNoMarginalProbabilityCalibrationModel() {
    return std::make_unique<NoMarginalProbabilityCalibrationModel>();
}

/**
 * A model for the calibration of joint probabilities that does not make any adjustments.
 */
class NoJointProbabilityCalibrationModel final : public INoJointProbabilityCalibrationModel {
    public:

        float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const override {
            return jointProbability;
        }
};

std::unique_ptr<IJointProbabilityCalibrationModel> NoJointProbabilityCalibrator::fitProbabilityCalibrationModel(
  const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return std::make_unique<NoJointProbabilityCalibrationModel>();
}

std::unique_ptr<IJointProbabilityCalibrationModel> NoJointProbabilityCalibrator::fitProbabilityCalibrationModel(
  const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return std::make_unique<NoJointProbabilityCalibrationModel>();
}

std::unique_ptr<IJointProbabilityCalibrationModel> NoJointProbabilityCalibrator::fitProbabilityCalibrationModel(
  BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return std::make_unique<NoJointProbabilityCalibrationModel>();
}

std::unique_ptr<IJointProbabilityCalibrationModel> NoJointProbabilityCalibrator::fitProbabilityCalibrationModel(
  BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return std::make_unique<NoJointProbabilityCalibrationModel>();
}

/**
 * A factory that allows to create instances of the class `NoJointProbabilityCalibrator`.
 */
class NoJointProbabilityCalibratorFactory final : public IJointProbabilityCalibratorFactory {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> create(const ILabelSpaceInfo& labelSpaceInfo) const override {
            return std::make_unique<NoJointProbabilityCalibrator>();
        }
};

bool NoJointProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
    return false;
}

bool NoJointProbabilityCalibratorConfig::isLabelVectorSetNeeded() const {
    return false;
}

std::unique_ptr<IJointProbabilityCalibratorFactory>
  NoJointProbabilityCalibratorConfig::createJointProbabilityCalibratorFactory() const {
    return std::make_unique<NoJointProbabilityCalibratorFactory>();
}

std::unique_ptr<INoJointProbabilityCalibrationModel> createNoJointProbabilityCalibrationModel() {
    return std::make_unique<NoJointProbabilityCalibrationModel>();
}
