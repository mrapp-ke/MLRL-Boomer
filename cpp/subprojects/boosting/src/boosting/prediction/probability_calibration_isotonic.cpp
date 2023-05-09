#include "boosting/prediction/probability_calibration_isotonic.hpp"

namespace boosting {

    template<typename IndexIterator, typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      IndexIterator indexIterator, uint32 numExamples, const LabelMatrix& labelMatrix, const IStatistics& statistics) {
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> calibrationModelPtr =
          std::make_unique<IsotonicMarginalProbabilityCalibrationModel>(numLabels);
        // TODO
        return calibrationModelPtr;
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const SinglePartition& partition, const LabelMatrix& labelMatrix, const IStatistics& statistics) {
        return fitMarginalProbabilityCalibrationModel(partition.cbegin(), partition.getNumElements(), labelMatrix,
                                                      statistics);
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const BiPartition& partition, uint32 useHoldoutSet, const LabelMatrix& labelMatrix,
      const IStatistics& statistics) {
        BiPartition::const_iterator indexIterator;
        uint32 numExamples;

        if (useHoldoutSet) {
            indexIterator = partition.second_cbegin();
            numExamples = partition.getNumSecond();
        } else {
            indexIterator = partition.first_cbegin();
            numExamples = partition.getNumFirst();
        }

        return fitMarginalProbabilityCalibrationModel(indexIterator, numExamples, labelMatrix, statistics);
    }

    /**
     * An implementation of the type `IMarginalProbabilityCalibrator` that does fit a model for the calibration of
     * marginal probabilities via isotonic regression.
     */
    class IsotonicMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
        private:

            bool useHoldoutSet_;

        public:

            /**
             * @brief Construct a new Isotonic Marginal Probability Calibrator object
             *
             * @param useHoldoutSet True, if the calibration model should be fit to the examples in the holdout set, if
             *                      available, false if the training set should be used instead
             */
            IsotonicMarginalProbabilityCalibrator(bool useHoldoutSet) : useHoldoutSet_(useHoldoutSet) {}

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics);
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
        return std::make_unique<IsotonicMarginalProbabilityCalibrator>(useHoldoutSet_);
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
                return createIsotonicJointProbabilityCalibrationModel();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return createIsotonicJointProbabilityCalibrationModel();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return createIsotonicJointProbabilityCalibrationModel();
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics,
              const IMarginalProbabilityCalibrationModel& IsotonicMarginalProbabilityCalibrationModel) const override {
                return createIsotonicJointProbabilityCalibrationModel();
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

    bool IsotonicJointProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
        return useHoldoutSet_;
    }

    std::unique_ptr<IJointProbabilityCalibrator>
      IsotonicJointProbabilityCalibratorConfig::createJointProbabilityCalibrator() const {
        return std::make_unique<IsotonicJointProbabilityCalibrator>();
    }
}
