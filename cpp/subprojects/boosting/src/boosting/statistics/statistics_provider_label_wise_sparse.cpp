#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "boosting/statistics/statistics_provider_label_wise_sparse.hpp"
#include "boosting/data/matrix_lil_numeric.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_label_wise.hpp"
#include "omp.h"


namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticMatrix final : public SparseLabelWiseStatisticView {

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            SparseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : SparseLabelWiseStatisticView(numCols, new LilMatrix<Tuple<float64>>(numRows, numCols)) {

            }

            ~SparseLabelWiseStatisticMatrix() {
                delete statistics_;
            }

    };


    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied label-wise and are stored using sparse data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class SparseLabelWiseStatistics final : public AbstractLabelWiseStatistics<LabelMatrix,
                                                                               DenseLabelWiseStatisticVector,
                                                                               SparseLabelWiseStatisticView,
                                                                               DenseLabelWiseStatisticMatrix,
                                                                               NumericLilMatrix<float64>,
                                                                               ISparseLabelWiseLoss,
                                                                               ISparseEvaluationMeasure,
                                                                               ILabelWiseRuleEvaluationFactory> {

        public:

            /**
             * @param lossPtr               An unique pointer to an object of template type `LossFunction` that
             *                              implements the loss function that should be used for calculating gradients
             *                              and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `ISparseEvaluationMeasure` that
             *                              implements the evaluation measure that should be used to assess the quality
             *                              of predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `ISparseLabelWiseRuleEvaluationFactory`,
             *                              that allows to create instances of the class that is used for calculating
             *                              the predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `SparseLabelWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericLilMatrix` that stores the
             *                              currently predicted scores
             */
            SparseLabelWiseStatistics(std::unique_ptr<ISparseLabelWiseLoss> lossPtr,
                                      std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
                                      const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                      const LabelMatrix& labelMatrix,
                                      std::unique_ptr<SparseLabelWiseStatisticView> statisticViewPtr,
                                      std::unique_ptr<NumericLilMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, SparseLabelWiseStatisticView,
                                              DenseLabelWiseStatisticMatrix, NumericLilMatrix<float64>,
                                              ISparseLabelWiseLoss, ISparseEvaluationMeasure,
                                              ILabelWiseRuleEvaluationFactory>(
                      std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                      std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {

            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
            const ISparseLabelWiseLossFactory& lossFactory,
            const ISparseEvaluationMeasureFactory& evaluationMeasureFactory,
            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<ISparseLabelWiseLoss> lossPtr = lossFactory.createSparseLabelWiseLoss();
        std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr =
            evaluationMeasureFactory.createSparseEvaluationMeasure();
        std::unique_ptr<SparseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<SparseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericLilMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericLilMatrix<float64>>(numExamples, numLabels);
        const ISparseLabelWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const LilMatrix<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        SparseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
        firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                  IndexIterator(labelMatrixPtr->getNumCols()), *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseLabelWiseStatistics<LabelMatrix>>(std::move(lossPtr),
                                                                        std::move(evaluationMeasurePtr),
                                                                        ruleEvaluationFactory, labelMatrix,
                                                                        std::move(statisticMatrixPtr),
                                                                        std::move(scoreMatrixPtr));
    }

    SparseLabelWiseStatisticsProviderFactory::SparseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr,
            std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {

    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CContiguousConstView<const uint8>& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                             numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const BinaryCsrConstView& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *regularRuleEvaluationFactoryPtr_,
                             numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

#ifdef _WIN32
    #pragma warning( pop )
#endif
