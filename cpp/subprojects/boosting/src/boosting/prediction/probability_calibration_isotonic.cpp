#include "boosting/prediction/probability_calibration_isotonic.hpp"

namespace boosting {

    /**
     * An implementation of the type `IMarginalProbabilityCalibrator` that does fit a model for the calibration of
     * marginal probabilities via isotonic regression.
     */
    class IsotonicMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
        public:

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
                return std::make_unique<IsotonicMarginalProbabilityCalibrationModel>();
            }
    };

    IsotonicMarginalProbabilityCalibratorConfig::IsotonicMarginalProbabilityCalibratorConfig() : useHoldoutSet_(true) {}

    bool IsotonicMarginalProbabilityCalibratorConfig::isHoldoutSetUsed() const {
        return useHoldoutSet_;
    }

    IIsotonicMarginalProbabilityCalibratorConfig& IsotonicMarginalProbabilityCalibratorConfig::setUseHoldoutSet(
      bool useHoldoutSet) {
        useHoldoutSet_ = useHoldoutSet;
        return *this;
    }

    bool IsotonicMarginalProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
        return useHoldoutSet_;
    }

    std::unique_ptr<IMarginalProbabilityCalibrator>
      IsotonicMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibrator() const {
        return std::make_unique<IsotonicMarginalProbabilityCalibrator>();
    }

    /**
     * An implementation of the type `IJointProbabilityCalibrator` that does fit a model for the calibration of joint
     * probabilities via isotonic regression.
     */
    class IsotonicJointProbabilityCalibrator final : public IJointProbabilityCalibrator {
        public:

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return std::make_unique<IsotonicJointProbabilityCalibrationModel>();
            }
    };

    IsotonicJointProbabilityCalibratorConfig::IsotonicJointProbabilityCalibratorConfig() : useHoldoutSet_(true) {}

    bool IsotonicJointProbabilityCalibratorConfig::isHoldoutSetUsed() const {
        return useHoldoutSet_;
    }

    IIsotonicJointProbabilityCalibratorConfig& IsotonicJointProbabilityCalibratorConfig::setUseHoldoutSet(
      bool useHoldoutSet) {
        useHoldoutSet_ = useHoldoutSet;
        return *this;
    }

    std::unique_ptr<IJointProbabilityCalibrator>
      IsotonicJointProbabilityCalibratorConfig::createJointProbabilityCalibrator() const {
        return std::make_unique<IsotonicJointProbabilityCalibrator>();
    }
}
