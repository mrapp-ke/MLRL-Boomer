#include "boosting/prediction/probability_calibration_isotonic.hpp"

#include "boosting/statistics/statistics.hpp"
#include "common/data/arrays.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/prediction/probability_calibration_no.hpp"

#include <algorithm>

namespace boosting {

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicMarginalProbabilityCalibrationModel& calibrationModel, const CContiguousLabelMatrix& labelMatrix,
      const CContiguousConstView<float64>& scoreMatrix,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
            CContiguousConstView<float64>::value_const_iterator scoreIterator =
              scoreMatrix.row_values_cbegin(exampleIndex);

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = labelIterator[j] ? 1 : 0;
                float64 score = scoreIterator[j];
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                calibrationModel.addBin(j, marginalProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicMarginalProbabilityCalibrationModel& calibrationModel, const CsrLabelMatrix& labelMatrix,
      const CContiguousConstView<float64>& scoreMatrix,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            auto labelIterator = make_binary_forward_iterator(labelMatrix.row_indices_cbegin(exampleIndex),
                                                              labelMatrix.row_indices_cend(exampleIndex));
            CContiguousConstView<float64>::value_const_iterator scoreIterator =
              scoreMatrix.row_values_cbegin(exampleIndex);

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = (*labelIterator) ? 1 : 0;
                float64 score = scoreIterator[j];
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                calibrationModel.addBin(j, marginalProbability, trueProbability);
                labelIterator++;
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicMarginalProbabilityCalibrationModel& calibrationModel, const CContiguousLabelMatrix& labelMatrix,
      const SparseSetMatrix<float64>& scoreMatrix, const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numLabels; i++) {
            calibrationModel.addBin(i, 0, 0);
        }

        uint32* numSparsePerLabel = new uint32[numLabels] {};

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
            SparseSetMatrix<float64>::const_row scoreRow = scoreMatrix[exampleIndex];

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = labelIterator[j] ? 1 : 0;
                const IndexedValue<float64>* entry = scoreRow[j];

                if (entry) {
                    float64 score = entry->value;
                    float64 marginalProbability =
                      marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                    calibrationModel.addBin(j, marginalProbability, trueProbability);
                } else {
                    IsotonicMarginalProbabilityCalibrationModel::bin_list bins = calibrationModel[j];
                    Tuple<float64>& firstBin = bins[0];
                    uint32 numSparse = numSparsePerLabel[j] + 1;

                    if (numSparse > 1) {
                        firstBin.second = iterativeArithmeticMean(numSparse, trueProbability, firstBin.second);
                    } else {
                        firstBin.second = trueProbability;
                    }

                    numSparsePerLabel[j] = numSparse;
                }
            }
        }

        delete[] numSparsePerLabel;
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicMarginalProbabilityCalibrationModel& calibrationModel, const CsrLabelMatrix& labelMatrix,
      const SparseSetMatrix<float64>& scoreMatrix, const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numLabels; i++) {
            calibrationModel.addBin(i, 0, 0);
        }

        uint32* numSparsePerLabel = new uint32[numLabels];
        setArrayToValue(numSparsePerLabel, numLabels, numExamples);
        uint32* numSparseRelevantPerLabel = new uint32[numLabels] {};

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CsrLabelMatrix::index_const_iterator labelIndicesBegin = labelMatrix.row_indices_cbegin(exampleIndex);
            CsrLabelMatrix::index_const_iterator labelIndicesEnd = labelMatrix.row_indices_cend(exampleIndex);
            uint32 numRelevantLabels = labelIndicesEnd - labelIndicesBegin;

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndicesBegin[j];
                numSparseRelevantPerLabel[labelIndex] += 1;
            }

            for (auto it = scoreMatrix.row_cbegin(exampleIndex); it != scoreMatrix.row_cend(exampleIndex); it++) {
                const IndexedValue<float64>& entry = *it;
                uint32 labelIndex = entry.index;
                float64 score = entry.value;
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(labelIndex, score);
                bool trueLabel = std::binary_search(labelIndicesBegin, labelIndicesEnd, labelIndex);
                calibrationModel.addBin(labelIndex, marginalProbability, trueLabel ? 1 : 0);
                numSparsePerLabel[labelIndex] -= 1;

                if (trueLabel) {
                    numSparseRelevantPerLabel[labelIndex] -= 1;
                }
            }
        }

        for (uint32 i = 0; i < numLabels; i++) {
            IsotonicMarginalProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            Tuple<float64>& firstBin = bins[0];
            firstBin.second = (float64) numSparseRelevantPerLabel[i] / (float64) numSparsePerLabel[i];
        }

        delete[] numSparsePerLabel;
        delete[] numSparseRelevantPerLabel;
    }

    static inline void eliminateDuplicateThresholds(IsotonicMarginalProbabilityCalibrationModel& calibrationModel,
                                                    uint32 numLabels) {
        for (uint32 i = 0; i < numLabels; i++) {
            // Sort bins in increasing order by their threshold...
            IsotonicMarginalProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            std::sort(bins.begin(), bins.end(), [=](const Tuple<float64>& lhs, const Tuple<float64>& rhs) {
                return lhs.first < rhs.first;
            });

            // Aggregate adjacent bins with identical thresholds by averaging their probabilities...
            uint32 numBins = (uint32) bins.size();
            uint32 previousIndex = 0;
            Tuple<float64> previousBin = bins[previousIndex];
            uint32 n = 0;

            for (uint32 j = 1; j < numBins; j++) {
                const Tuple<float64>& currentBin = bins[j];

                if (currentBin.first != previousBin.first) {
                    bins[n] = previousBin;
                    n++;
                    previousIndex = j;
                    previousBin = currentBin;
                } else {
                    uint32 numAggregated = j - previousIndex + 1;
                    previousBin.second = iterativeArithmeticMean(numAggregated, currentBin.second, previousBin.second);
                }
            }

            bins[n] = bins[numBins - 1];
            n++;
            bins.resize(n);
        }
    }

    template<typename IndexIterator, typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      IndexIterator indexIterator, uint32 numExamples, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunctionFactory& marginalProbabilityFunctionFactory) {
        // Create probability function...
        std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalProbabilityCalibrationModelPtr =
          createNoMarginalProbabilityCalibrationModel();
        std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr =
          marginalProbabilityFunctionFactory.create(*marginalProbabilityCalibrationModelPtr);

        // Extract thresholds and ground truth probabilities from score matrix and label matrix, respectively...
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> calibrationModelPtr =
          std::make_unique<IsotonicMarginalProbabilityCalibrationModel>(numLabels);
        const IBoostingStatistics& boostingStatistics = dynamic_cast<const IBoostingStatistics&>(statistics);
        auto denseVisitor =
          [=, &marginalProbabilityFunctionPtr, &calibrationModelPtr](const CContiguousConstView<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, numLabels, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, *marginalProbabilityFunctionPtr);
        };
        auto sparseVisitor =
          [=, &calibrationModelPtr, &marginalProbabilityFunctionPtr](const SparseSetMatrix<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, numLabels, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, *marginalProbabilityFunctionPtr);
        };
        boostingStatistics.visitScoreMatrix(denseVisitor, sparseVisitor);

        // Eliminate duplicate thresholds...
        eliminateDuplicateThresholds(*calibrationModelPtr, numLabels);

        // TODO Perform isotonic regression...

        return calibrationModelPtr;
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const SinglePartition& partition, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunctionFactory& marginalProbabilityFunctionFactory) {
        return fitMarginalProbabilityCalibrationModel(partition.cbegin(), partition.getNumElements(), labelMatrix,
                                                      statistics, marginalProbabilityFunctionFactory);
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const BiPartition& partition, uint32 useHoldoutSet, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunctionFactory& marginalProbabilityFunctionFactory) {
        BiPartition::const_iterator indexIterator;
        uint32 numExamples;

        if (useHoldoutSet) {
            indexIterator = partition.second_cbegin();
            numExamples = partition.getNumSecond();
        } else {
            indexIterator = partition.first_cbegin();
            numExamples = partition.getNumFirst();
        }

        return fitMarginalProbabilityCalibrationModel(indexIterator, numExamples, labelMatrix, statistics,
                                                      marginalProbabilityFunctionFactory);
    }

    /**
     * An implementation of the type `IMarginalProbabilityCalibrator` that does fit a model for the calibration of
     * marginal probabilities via isotonic regression.
     */
    class IsotonicMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
        private:

            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

            bool useHoldoutSet_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the transformation function to be used to
             *                                              transform regression scores that are predicted for
             *                                              individual labels into probabilities
             * @param useHoldoutSet                         True, if the calibration model should be fit to the examples
             *                                              in the holdout set, if available, false otherwise
             */
            IsotonicMarginalProbabilityCalibrator(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr,
              bool useHoldoutSet)
                : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)),
                  useHoldoutSet_(useHoldoutSet) {}

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionFactoryPtr_);
            }
    };

    IsotonicMarginalProbabilityCalibratorConfig::IsotonicMarginalProbabilityCalibratorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : useHoldoutSet_(true), lossConfigPtr_(lossConfigPtr) {}

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
        std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createMarginalProbabilityFunctionFactory();
        return std::make_unique<IsotonicMarginalProbabilityCalibrator>(std::move(marginalProbabilityFunctionFactoryPtr),
                                                                       useHoldoutSet_);
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
