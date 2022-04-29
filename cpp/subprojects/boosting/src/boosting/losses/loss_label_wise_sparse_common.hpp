#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "loss_label_wise_common.hpp"
#include <iostream>  // TODO Remove


namespace boosting {

    /**
     * An implementation of the type `ISparseLabelWiseLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluation the predictions for an individual
     * label, respectively.
     */
    class SparseLabelWiseLoss final : public LabelWiseLoss, public ISparseLabelWiseLoss {

        public:

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            SparseLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : LabelWiseLoss(updateFunction, evaluateFunction) {

            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::updateLabelWiseStatistics(CContiguousConstView, CompleteIndexVector)\n";
                std::exit(-1);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::updateLabelWiseStatistics(CContiguousConstView, PartialIndexVector)\n";
                std::exit(-1);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::updateLabelWiseStatistics(BinaryCsrConstView, CompleteIndexVector)\n";
                std::exit(-1);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const LilMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::updateLabelWiseStatistics(BinaryCsrConstView, PartialIndexVector)\n";
                std::exit(-1);
            }

            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::evaluate(CContiguousConstView)\n";
                std::exit(-1);
                return 0;
            }

            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const LilMatrix<float64>& scoreMatrix) const override {
                // TODO Implement
                std::cout << "SparseLabelWiseLoss::evaluate(BinaryCsrConstView)\n";
                std::exit(-1);
                return 0;
            }

    };

}
