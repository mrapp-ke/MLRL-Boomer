#include "mlrl/seco/statistics/statistics_provider_decomposable_sparse.hpp"

#include "mlrl/common/data/matrix_c_contiguous.hpp"
#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"
#include "mlrl/seco/data/view_statistic_decomposable_sparse.hpp"
#include "statistics_decomposable_common.hpp"
#include "statistics_provider_decomposable.hpp"

#include <type_traits>

namespace seco {

    template<typename IndexIterator>
    static inline void eraseIfPresent(SparseDecomposableStatisticView::row row, IndexIterator& iterator, uint32 index) {
        auto end = row.end();

        while (iterator != end && *iterator < index) {
            iterator++;
        }

        if (iterator != end && *iterator == index) {
            iterator = row.erase(iterator);
        }
    }

    template<typename IndexIterator>
    static inline void eraseIfPresent(SparseDecomposableStatisticView::row in, SparseDecomposableStatisticView::row ip,
                                      SparseDecomposableStatisticView::row rn, SparseDecomposableStatisticView::row rp,
                                      IndexIterator& inIterator, IndexIterator& ipIterator, IndexIterator& rnIterator,
                                      IndexIterator& rpIterator, bool trueLabel, bool prediction, uint32 index) {
        if (trueLabel) {
            if (prediction) {
                eraseIfPresent(rp, rpIterator, index);
            } else {
                eraseIfPresent(rn, rnIterator, index);
            }
        } else {
            if (prediction) {
                eraseIfPresent(ip, ipIterator, index);
            } else {
                eraseIfPresent(in, inIterator, index);
            }
        }
    }

    static inline uint32 increaseCoverageInternally(uint32 row, const CContiguousView<const uint8>& labelMatrix,
                                                    CContiguousView<uint32>& coverageMatrix,
                                                    SparseDecomposableStatisticView& statisticView,
                                                    PartialIndexVector::const_iterator indexIterator,
                                                    View<uint8>::const_iterator predictionIterator, uint32 numIndices) {
        auto groundTruthIterator = labelMatrix.values_cbegin(row);
        auto coverageIterator = coverageMatrix.values_begin(row);
        auto& in = statisticView.in_row(row);
        auto& ip = statisticView.ip_row(row);
        auto& rn = statisticView.rn_row(row);
        auto& rp = statisticView.rp_row(row);
        auto inIterator = in.begin();
        auto ipIterator = ip.begin();
        auto rnIterator = rn.begin();
        auto rpIterator = rp.begin();
        uint32 numModified = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 0) {
                bool trueLabel = groundTruthIterator[index];
                bool prediction = predictionIterator[i];
                eraseIfPresent(in, ip, rn, rp, inIterator, ipIterator, rnIterator, rpIterator, trueLabel, prediction,
                               index);

                if (prediction == trueLabel) {
                    numModified++;
                }
            }

            coverageIterator[index] = coverage + 1;
        }

        return numModified;
    }

    static inline uint32 increaseCoverageInternally(uint32 row, const BinaryCsrView& labelMatrix,
                                                    CContiguousView<uint32>& coverageMatrix,
                                                    SparseDecomposableStatisticView& statisticView,
                                                    PartialIndexVector::const_iterator indexIterator,
                                                    View<uint8>::const_iterator predictionIterator, uint32 numIndices) {
        auto groundTruthIterator =
          createBinarySparseForwardIterator(labelMatrix.indices_cbegin(row), labelMatrix.indices_cend(row));
        auto coverageIterator = coverageMatrix.values_begin(row);
        auto& in = statisticView.in_row(row);
        auto& ip = statisticView.ip_row(row);
        auto& rn = statisticView.rn_row(row);
        auto& rp = statisticView.rp_row(row);
        auto inIterator = in.begin();
        auto ipIterator = ip.begin();
        auto rnIterator = rn.begin();
        auto rpIterator = rp.begin();
        uint32 numModified = 0;
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 0) {
                std::advance(groundTruthIterator, index - previousIndex);
                bool trueLabel = *groundTruthIterator;
                bool prediction = predictionIterator[i];
                eraseIfPresent(in, ip, rn, rp, inIterator, ipIterator, rnIterator, rpIterator, trueLabel, prediction,
                               index);

                if (prediction == trueLabel) {
                    numModified++;
                }

                previousIndex = index;
            }

            coverageIterator[index] = coverage + 1;
        }

        return numModified;
    }

    template<typename IndexIterator>
    static inline void addIfNotPresent(SparseDecomposableStatisticView::row row, IndexIterator& iterator,
                                       uint32 index) {
        auto end = row.end();

        while (iterator != end && *iterator < index) {
            iterator++;
        }

        if (iterator == end || *iterator > index) {
            iterator = row.emplace(iterator, index);
        }
    }

    template<typename IndexIterator>
    static inline void addIfNotPresent(SparseDecomposableStatisticView::row in, SparseDecomposableStatisticView::row ip,
                                       SparseDecomposableStatisticView::row rn, SparseDecomposableStatisticView::row rp,
                                       IndexIterator& inIterator, IndexIterator& ipIterator, IndexIterator& rnIterator,
                                       IndexIterator& rpIterator, bool trueLabel, bool prediction, uint32 index) {
        if (trueLabel) {
            if (prediction) {
                addIfNotPresent(rp, rpIterator, index);
            } else {
                addIfNotPresent(rn, rnIterator, index);
            }
        } else {
            if (prediction) {
                addIfNotPresent(ip, ipIterator, index);
            } else {
                addIfNotPresent(in, inIterator, index);
            }
        }
    }

    static inline uint32 decreaseCoverageInternally(uint32 row, const CContiguousView<const uint8>& labelMatrix,
                                                    CContiguousView<uint32>& coverageMatrix,
                                                    SparseDecomposableStatisticView& statisticView,
                                                    PartialIndexVector::const_iterator indexIterator,
                                                    View<uint8>::const_iterator predictionIterator, uint32 numIndices) {
        auto groundTruthIterator = labelMatrix.values_cbegin(row);
        auto coverageIterator = coverageMatrix.values_begin(row);
        auto& in = statisticView.in_row(row);
        auto& ip = statisticView.ip_row(row);
        auto& rn = statisticView.rn_row(row);
        auto& rp = statisticView.rp_row(row);
        auto inIterator = in.begin();
        auto ipIterator = ip.begin();
        auto rnIterator = rn.begin();
        auto rpIterator = rp.begin();
        uint32 numModified = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 1) {
                bool trueLabel = groundTruthIterator[index];
                bool prediction = predictionIterator[i];
                addIfNotPresent(in, ip, rn, rp, inIterator, ipIterator, rnIterator, rpIterator, trueLabel, prediction,
                                index);

                if (prediction == trueLabel) {
                    numModified++;
                }
            }

            if (coverage > 0) {
                coverageIterator[index] = coverage - 1;
            }
        }

        return numModified;
    }

    static inline uint32 decreaseCoverageInternally(uint32 row, const BinaryCsrView& labelMatrix,
                                                    CContiguousView<uint32>& coverageMatrix,
                                                    SparseDecomposableStatisticView& statisticView,
                                                    PartialIndexVector::const_iterator indexIterator,
                                                    View<uint8>::const_iterator predictionIterator, uint32 numIndices) {
        auto groundTruthIterator =
          createBinarySparseForwardIterator(labelMatrix.indices_cbegin(row), labelMatrix.indices_cend(row));
        auto coverageIterator = coverageMatrix.values_begin(row);
        auto& in = statisticView.in_row(row);
        auto& ip = statisticView.ip_row(row);
        auto& rn = statisticView.rn_row(row);
        auto& rp = statisticView.rp_row(row);
        auto inIterator = in.begin();
        auto ipIterator = ip.begin();
        auto rnIterator = rn.begin();
        auto rpIterator = rp.begin();
        uint32 numModified = 0;
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 1) {
                std::advance(groundTruthIterator, index - previousIndex);
                bool trueLabel = *groundTruthIterator;
                bool prediction = predictionIterator[i];
                addIfNotPresent(in, ip, rn, rp, inIterator, ipIterator, rnIterator, rpIterator, trueLabel, prediction,
                                index);

                if (trueLabel == prediction) {
                    numModified++;
                }

                previousIndex = index;
            }

            if (coverage > 0) {
                coverageIterator[index] = coverage - 1;
            }
        }

        return numModified;
    }

    static inline uint32 initializeMajorityLabelVector(const CContiguousView<const uint8>& labelMatrix,
                                                       ResizableBinarySparseArrayVector& majorityLabelVector) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        float64 threshold = numExamples / 2.0;
        auto majorityIterator = majorityLabelVector.begin();
        uint32 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrix.values_cbegin(j)[i];
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;  // rp-pairs (minority class)
            }
        }

        majorityLabelVector.setNumElements(n, true);
        return sumOfUncoveredWeights;
    }

    static inline uint32 initializeMajorityLabelVector(const BinaryCsrView& labelMatrix,
                                                       ResizableBinarySparseArrayVector& majorityLabelVector) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;
        auto majorityIterator = majorityLabelVector.begin();
        std::fill(majorityIterator, majorityLabelVector.end(), 0);

        for (uint32 i = 0; i < numExamples; i++) {
            auto indexIterator = labelMatrix.indices_cbegin(i);
            uint32 numElements = labelMatrix.indices_cend(i) - indexIterator;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                majorityIterator[index] += 1;
            }
        }

        float64 threshold = numExamples / 2.0;
        uint32 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = majorityIterator[i];

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVector.setNumElements(n, true);
        return sumOfUncoveredWeights;
    }

    static inline void initializeStatisticMatrix(const CContiguousView<const uint8>& labelMatrix,
                                                 const ResizableBinarySparseArrayVector& majorityLabelVector,
                                                 SparseDecomposableStatisticView& statisticView) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;

        for (uint32 i = 0; i < numExamples; i++) {
            auto& in = statisticView.in_row(i);
            auto& ip = statisticView.ip_row(i);
            auto& rn = statisticView.rn_row(i);
            auto& rp = statisticView.rp_row(i);
            auto groundTruthIterator = labelMatrix.values_cbegin(i);
            auto majorityIterator =
              createBinarySparseForwardIterator(majorityLabelVector.cbegin(), majorityLabelVector.cend());

            for (uint32 j = 0; j < numLabels; j++) {
                bool trueLabel = groundTruthIterator[j];
                bool majorityLabel = *majorityIterator;
                bool prediction = !majorityLabel;  // Rules predict the opposite of the majority label

                if (trueLabel) {
                    if (prediction) {
                        rp.emplace_back(j);
                    } else {
                        rn.emplace_back(j);
                    }
                } else {
                    if (prediction) {
                        ip.emplace_back(j);
                    } else {
                        in.emplace_back(j);
                    }
                }

                majorityIterator++;
            }
        }
    }

    static inline void initializeStatisticMatrix(const BinaryCsrView& labelMatrix,
                                                 const ResizableBinarySparseArrayVector& majorityLabelVector,
                                                 SparseDecomposableStatisticView& statisticView) {
        uint32 numExamples = labelMatrix.numRows;
        uint32 numLabels = labelMatrix.numCols;

        for (uint32 i = 0; i < numExamples; i++) {
            auto& in = statisticView.in_row(i);
            auto& ip = statisticView.ip_row(i);
            auto& rn = statisticView.rn_row(i);
            auto& rp = statisticView.rp_row(i);
            auto groundTruthIterator =
              createBinarySparseForwardIterator(labelMatrix.indices_cbegin(i), labelMatrix.indices_cend(i));
            auto majorityIterator =
              createBinarySparseForwardIterator(majorityLabelVector.cbegin(), majorityLabelVector.cend());

            for (uint32 j = 0; j < numLabels; j++) {
                bool trueLabel = *groundTruthIterator;
                bool majorityLabel = *majorityIterator;
                bool prediction = !majorityLabel;  // Rules predict the opposite of the majority label

                if (trueLabel) {
                    if (prediction) {
                        rp.emplace_back(j);
                    } else {
                        rn.emplace_back(j);
                    }
                } else {
                    if (prediction) {
                        ip.emplace_back(j);
                    } else {
                        in.emplace_back(j);
                    }
                }

                groundTruthIterator++;
                majorityIterator++;
            }
        }
    }

    /**
     * A matrix that stores confusion matrix elements using a sparse matrix in the list of lists (LIL) format.
     *
     * @tparam LabelMatrix  The type of the label matrix that provides random or row-wise access to the labels of the
     *                      training examples
     * @tparam VectorMath   The type that implements basic operations for calculating with numerical arrays
     */
    template<typename LabelMatrix, typename VectorMath>
    class SparseDecomposableStatisticMatrix final
        : public ClearableViewDecorator<MatrixDecorator<SparseDecomposableStatisticView>> {
        private:

            ResizableBinarySparseArrayVector majorityLabelVector_;

            CContiguousMatrix<uint32> coverageMatrix_;

            const LabelMatrix& labelMatrix_;

            uint32 sumOfUncoveredWeights_;

        public:

            /**
             * @param labelMatrix A reference to an object of template type `LabelMatrix` that provides random or
             *                    row-wise access to the labels of the training examples
             */
            SparseDecomposableStatisticMatrix(const LabelMatrix& labelMatrix)
                : ClearableViewDecorator<MatrixDecorator<SparseDecomposableStatisticView>>(
                    SparseDecomposableStatisticView(labelMatrix.numRows, labelMatrix.numCols)),
                  majorityLabelVector_(labelMatrix.numCols),
                  coverageMatrix_(labelMatrix.numRows, labelMatrix.numCols, true), labelMatrix_(labelMatrix) {
                sumOfUncoveredWeights_ = initializeMajorityLabelVector(labelMatrix_, majorityLabelVector_);
                initializeStatisticMatrix(labelMatrix_, majorityLabelVector_, this->getView());
            }

            /**
             * Increases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for a subset of the available labels.
             *
             * @param row             The row
             * @param predictionBegin An iterator to the beginning of the predictions
             * @param predictionEnd   An iterator to the end of the predictions
             * @param indicesBegin    An iterator to the beginning of the label indices
             * @param indicesEnd      An iterator to the end of the label indices
             */
            void increaseCoverage(uint32 row, View<uint8>::const_iterator predictionsBegin,
                                  View<uint8>::const_iterator predictionsEnd,
                                  PartialIndexVector::const_iterator indicesBegin,
                                  PartialIndexVector::const_iterator indicesEnd) {
                uint32 numIndices = indicesEnd - indicesBegin;
                sumOfUncoveredWeights_ -=
                  increaseCoverageInternally(row, labelMatrix_, coverageMatrix_.getView(), this->getView(),
                                             indicesBegin, predictionsBegin, numIndices);
            }

            /**
             * Decreases the number of times the elements at a specific row of this matrix are covered, given the
             * predictions of a rule that predicts for a subset of the available labels.
             *
             * @param row             The row
             * @param predictionBegin An iterator to the beginning of the predictions
             * @param predictionEnd   An iterator to the end of the predictions
             * @param indicesBegin    An iterator to the beginning of the label indices
             * @param indicesEnd      An iterator to the end of the label indices
             */
            void decreaseCoverage(uint32 row, View<uint8>::const_iterator predictionBegin,
                                  View<uint8>::const_iterator predictionEnd,
                                  PartialIndexVector::const_iterator indicesBegin,
                                  PartialIndexVector::const_iterator indicesEnd) {
                uint32 numIndices = indicesEnd - indicesBegin;
                sumOfUncoveredWeights_ +=
                  decreaseCoverageInternally(row, labelMatrix_, coverageMatrix_.getView(), this->getView(),
                                             indicesBegin, predictionBegin, numIndices);
            }

            /**
             * Returns an `index_const_iterator` to the beginning of the predictions of the default rule.
             *
             * @return An `index_const_iterator` to the beginning
             */
            BinarySparseArrayVector::const_iterator majority_label_indices_cbegin() const {
                return majorityLabelVector_.cbegin();
            }

            /**
             * Returns an `index_const_iterator` to the end of the predictions of the default rule.
             *
             * @return An `index_const_iterator` to the end
             */
            BinarySparseArrayVector::const_iterator majority_label_indices_cend() const {
                return majorityLabelVector_.cend();
            }

            /**
             * Returns the sum of the weights of all examples and labels that have not been covered yet.
             *
             * @return The sum of the weights of all examples and labels that have not been covered yet
             */
            uint32 getSumOfUncoveredWeights() const {
                return sumOfUncoveredWeights_;
            }
    };

    /**
     * Provides access to the elements of confusion matrices that are computed independently for each output and are
     * stored using sparse data structures.
     *
     * @tparam LabelMatrix  The type of the matrix that provides access to the labels of the training examples
     * @tparam VectorMath   The type that implements basic operations for calculating with numerical arrays
     */
    template<typename LabelMatrix, typename VectorMath>
    class SparseDecomposableStatistics final
        : public AbstractDecomposableStatistics<SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>,
                                                IDecomposableRuleEvaluationFactory> {
        private:

            using StatisticMatrix = SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>;

            template<typename StatisticType>
            using StatisticVector = DenseConfusionMatrixVector<StatisticType, VectorMath>;

            template<typename WeightVector, typename IndexVector, typename StatisticType>
            using StatisticsSubset = CoverageStatisticsSubset<
              CoverageStatisticsState<SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>>,
              StatisticVector<StatisticType>, WeightVector, IndexVector, IDecomposableRuleEvaluationFactory>;

            template<typename WeightVector, typename StatisticType>
            using WeightedStatistics =
              WeightedStatistics<CoverageStatisticsState<SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>>,
                                 StatisticVector<StatisticType>, WeightVector, IDecomposableRuleEvaluationFactory>;

        public:

            /**
             * @param labelMatrix             A reference to an object of template type `LabelMatrix` that provides
             *                                access to the labels of the training examples
             * @param ruleEvaluationFactory   A reference to an object of type `IDecomposableRuleEvaluationFactory` that
             *                                allows to create instances of the class that is used for calculating the
             *                                predictions of rules, as well as their overall quality
             */
            SparseDecomposableStatistics(const LabelMatrix& labelMatrix,
                                         const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory)
                : AbstractDecomposableStatistics<SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>,
                                                 IDecomposableRuleEvaluationFactory>(
                    std::make_unique<SparseDecomposableStatisticMatrix<LabelMatrix, VectorMath>>(labelMatrix),
                    ruleEvaluationFactory) {}

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                auto subsetSumVectorPtr = std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<EqualWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const EqualWeightVector& weights) const override {
                auto subsetSumVectorPtr = std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<EqualWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                auto subsetSumVectorPtr = std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<BitWeightVector, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const BitWeightVector& weights) const override {
                auto subsetSumVectorPtr = std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<BitWeightVector, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<uint16>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<uint16>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                std::unique_ptr<StatisticVector<float32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<float32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, CompleteIndexVector, float32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& outputIndices,
                                                            const DenseWeightVector<float32>& weights) const override {
                std::unique_ptr<StatisticVector<float32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<float32>>(this->getNumOutputs(), true);
                return std::make_unique<StatisticsSubset<DenseWeightVector<float32>, PartialIndexVector, float32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<EqualWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<EqualWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<BitWeightVector>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<BitWeightVector>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<uint16>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<uint16>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const CompleteIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, CompleteIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createSubset`
             */
            std::unique_ptr<IStatisticsSubset> createSubset(
              const PartialIndexVector& outputIndices,
              const OutOfSampleWeightVector<DenseWeightVector<float32>>& weights) const override {
                std::unique_ptr<StatisticVector<uint32>> subsetSumVectorPtr =
                  std::make_unique<StatisticVector<uint32>>(this->getNumOutputs(), true);
                return std::make_unique<
                  StatisticsSubset<OutOfSampleWeightVector<DenseWeightVector<float32>>, PartialIndexVector, uint32>>(
                  *this->statePtr_, weights, outputIndices, *this->ruleEvaluationFactory_,
                  std::move(subsetSumVectorPtr));
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const EqualWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<EqualWeightVector, uint32>>(*this->statePtr_, weights,
                                                                                       *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const BitWeightVector& weights) const override {
                return std::make_unique<WeightedStatistics<BitWeightVector, uint32>>(*this->statePtr_, weights,
                                                                                     *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<uint16>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<uint16>, uint32>>(
                  *this->statePtr_, weights, *this->ruleEvaluationFactory_);
            }

            /**
             * @see `IStatistics::createWeightedStatistics`
             */
            std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
              const DenseWeightVector<float32>& weights) const override {
                return std::make_unique<WeightedStatistics<DenseWeightVector<float32>, float32>>(
                  *this->statePtr_, weights, *this->ruleEvaluationFactory_);
            }
    };

    template<typename VectorMath>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const CContiguousView<const uint8>& labelMatrix,
      std::type_identity<VectorMath>) {
        return std::make_unique<SparseDecomposableStatistics<CContiguousView<const uint8>, VectorMath>>(
          labelMatrix, ruleEvaluationFactory);
    }

    template<typename VectorMath>
    static inline std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> createStatistics(
      const IDecomposableRuleEvaluationFactory& ruleEvaluationFactory, const BinaryCsrView& labelMatrix,
      std::type_identity<VectorMath> vectorMath) {
        return std::make_unique<SparseDecomposableStatistics<BinaryCsrView, VectorMath>>(labelMatrix,
                                                                                         ruleEvaluationFactory);
    }

    template<typename VectorMath>
    SparseDecomposableStatisticsProviderFactory<VectorMath>::SparseDecomposableStatisticsProviderFactory(
      std::unique_ptr<IDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)) {}

    template<typename VectorMath>
    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory<VectorMath>::create(
      const CContiguousView<const uint8>& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template<typename VectorMath>
    std::unique_ptr<IStatisticsProvider> SparseDecomposableStatisticsProviderFactory<VectorMath>::create(
      const BinaryCsrView& labelMatrix) const {
        std::unique_ptr<IDecomposableStatistics<IDecomposableRuleEvaluationFactory>> statisticsPtr =
          createStatistics(*defaultRuleEvaluationFactoryPtr_, labelMatrix, std::type_identity<VectorMath> {});
        return std::make_unique<DecomposableStatisticsProvider<IDecomposableRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    template class SparseDecomposableStatisticsProviderFactory<SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class SparseDecomposableStatisticsProviderFactory<SimdVectorMath>;
#endif
}
