#include "boosting/statistics/statistics_provider_factory_label_wise_sparse.hpp"
#include "boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"
#include "boosting/data/matrix_lil_numeric.hpp"
#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "common/measures/measure_evaluation_sparse.hpp"
#include "common/validation.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_provider_label_wise.hpp"


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
                : SparseLabelWiseStatisticView(new LilMatrix<Tuple<float64>>(numRows), numCols) {

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
                                                                               SparseLabelWiseStatisticVector,
                                                                               SparseLabelWiseStatisticView,
                                                                               SparseLabelWiseStatisticMatrix,
                                                                               NumericLilMatrix<float64>,
                                                                               ISparseLabelWiseLoss,
                                                                               ISparseEvaluationMeasure,
                                                                               ISparseLabelWiseRuleEvaluationFactory> {

        public:

            /**
             * @param lossFunction          A reference to an object of type `ISparseLabelWiseLoss`, representing the
             *                              loss function to be used for calculating gradients and Hessians
             * @param evaluationMeasure     A reference to an object of type `ISparseEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions for a specific statistic
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
            SparseLabelWiseStatistics(const ISparseLabelWiseLoss& lossFunction,
                                      const ISparseEvaluationMeasure& evaluationMeasure,
                                      const ISparseLabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                      const LabelMatrix& labelMatrix,
                                      std::unique_ptr<SparseLabelWiseStatisticView> statisticViewPtr,
                                      std::unique_ptr<NumericLilMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, SparseLabelWiseStatisticVector, SparseLabelWiseStatisticView,
                                              SparseLabelWiseStatisticMatrix, NumericLilMatrix<float64>,
                                              ISparseLabelWiseLoss, ISparseEvaluationMeasure,
                                              ISparseLabelWiseRuleEvaluationFactory>(
                      lossFunction, evaluationMeasure, ruleEvaluationFactory, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> createStatistics(
            const ISparseLabelWiseLoss& lossFunction, const ISparseEvaluationMeasure& evaluationMeasure,
            const ISparseLabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<SparseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<SparseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericLilMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericLilMatrix<float64>>(numExamples);
        const ISparseLabelWiseLoss* lossFunctionRawPtr = &lossFunction;
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const LilMatrix<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        SparseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                          IndexIterator(labelMatrixPtr->getNumCols()),
                                                          *statisticMatrixRawPtr);
        }

        return std::make_unique<SparseLabelWiseStatistics<LabelMatrix>>(lossFunction, evaluationMeasure,
                                                                        ruleEvaluationFactory, labelMatrix,
                                                                        std::move(statisticMatrixPtr),
                                                                        std::move(scoreMatrixPtr));
    }

    SparseLabelWiseStatisticsProviderFactory::SparseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ISparseLabelWiseLoss> lossFunctionPtr,
            std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
            std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("evaluationMeasurePtr", evaluationMeasurePtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFunctionPtr_, *evaluationMeasurePtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_,
                             labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ISparseLabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ISparseLabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFunctionPtr_, *evaluationMeasurePtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_,
                             labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ISparseLabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}
