#include "boosting/statistics/statistics_provider_factory_label_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "common/validation.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_label_wise_provider.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function using C-contiguous arrays.
     */
    class DenseLabelWiseStatisticMatrix final : public DenseLabelWiseStatisticView {

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : DenseLabelWiseStatisticView(numRows, numCols,
                                              (Tuple<float64>*) malloc(numRows * numCols * sizeof(Tuple<float64>))) {

            }

            ~DenseLabelWiseStatisticMatrix() {
                free(statistics_);
            }

    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied label-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final : public AbstractLabelWiseStatistics<LabelMatrix,
                                                                              DenseLabelWiseStatisticVector,
                                                                              DenseLabelWiseStatisticView,
                                                                              DenseLabelWiseStatisticMatrix,
                                                                              NumericDenseMatrix<float64>,
                                                                              ILabelWiseLoss> {

        public:

            /**
             * @param lossFunction          A reference to an object of type `ILabelWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseLabelWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericDenseMatrix` that stores the
             *                              currently predicted scores
             */
            DenseLabelWiseStatistics(const ILabelWiseLoss& lossFunction,
                                     const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                     const LabelMatrix& labelMatrix,
                                     std::unique_ptr<DenseLabelWiseStatisticView> statisticViewPtr,
                                     std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticView,
                                              DenseLabelWiseStatisticMatrix, NumericDenseMatrix<float64>,
                                              ILabelWiseLoss>(
                      lossFunction, ruleEvaluationFactory, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics> createInternally(
            const ILabelWiseLoss& lossFunction, const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
            uint32 numThreads, const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericDenseMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossFunctionRawPtr = &lossFunction;
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                          IndexIterator(labelMatrixPtr->getNumCols()),
                                                          *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(lossFunction, ruleEvaluationFactory, labelMatrix,
                                                                       std::move(statisticMatrixPtr),
                                                                       std::move(scoreMatrixPtr));
    }

    /**
     * A factory that allows to create new instances of the class `ILabelWiseStatistics` that use dense data structures
     * to store the statistics.
     */
    class DenseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

        private:

            const ILabelWiseLoss& lossFunction_;

            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunction          A reference to an object of type `ILabelWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                              allows to create instances of the class that is used to calculate the
             *                              predictions, as well as corresponding quality scores, of rules
             * @param numThreads            The number of CPU threads to be used to calculate the initial statistics in
             *                              parallel. Must be at least 1
             */
            DenseLabelWiseStatisticsFactory(const ILabelWiseLoss& lossFunction,
                                            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                            uint32 numThreads)
                : lossFunction_(lossFunction), ruleEvaluationFactory_(ruleEvaluationFactory), numThreads_(numThreads) {

            }

            std::unique_ptr<ILabelWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override {
                return createInternally<CContiguousLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_,
                                                                labelMatrix);
            }

            std::unique_ptr<ILabelWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override {
                return createInternally<CsrLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_,
                                                        labelMatrix);
            }

    };

    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                          numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                          numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }


}
