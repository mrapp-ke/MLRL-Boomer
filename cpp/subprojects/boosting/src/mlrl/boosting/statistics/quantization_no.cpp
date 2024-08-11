#include "mlrl/boosting/statistics/quantization_no.hpp"

namespace boosting {

    template<typename View>
    class NoQuantizationMatrix final : public IQuantizationMatrix<View> {
        private:

            const View& view_;

        public:

            NoQuantizationMatrix(const View& view) : view_(view) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            const typename IQuantizationMatrix<View>::view_type& getView() const override {
                return view_;
            }
    };

    class NoDenseDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<CContiguousView<Tuple<float64>>>> quantizationMatrixPtr_;

        public:

            NoDenseDecomposableQuantization(const CContiguousView<Tuple<float64>>& view)
                : quantizationMatrixPtr_(
                    std::make_unique<NoQuantizationMatrix<CContiguousView<Tuple<float64>>>>(view)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor denseDecomposableMatrixVisitor,
              IQuantization::SparseDecomposableMatrixVisitor sparseDecomposableMatrixVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor denseNonDecomposableMatrixVisitor) override {
                denseDecomposableMatrixVisitor(quantizationMatrixPtr_);
            }
    };

    class NoSparseDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<SparseSetView<Tuple<float64>>>> quantizationMatrixPtr_;

        public:

            NoSparseDecomposableQuantization(const SparseSetView<Tuple<float64>>& view)
                : quantizationMatrixPtr_(std::make_unique<NoQuantizationMatrix<SparseSetView<Tuple<float64>>>>(view)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor denseDecomposableMatrixVisitor,
              IQuantization::SparseDecomposableMatrixVisitor sparseDecomposableMatrixVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor denseNonDecomposableMatrixVisitor) override {
                sparseDecomposableMatrixVisitor(quantizationMatrixPtr_);
            }
    };

    class NoDenseNonDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView>> quantizationMatrixPtr_;

        public:

            NoDenseNonDecomposableQuantization(const DenseNonDecomposableStatisticView& view)
                : quantizationMatrixPtr_(
                    std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView>>(view)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor denseDecomposableMatrixVisitor,
              IQuantization::SparseDecomposableMatrixVisitor sparseDecomposableMatrixVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor denseNonDecomposableMatrixVisitor) override {
                denseNonDecomposableMatrixVisitor(quantizationMatrixPtr_);
            }
    };

    class NoQuantizationFactory final : public IQuantizationFactory {
        public:

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization>(statisticMatrix);
            }
    };

    std::unique_ptr<IQuantizationFactory> NoQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<NoQuantizationFactory>();
    }

}
