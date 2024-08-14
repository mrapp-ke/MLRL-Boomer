#include "mlrl/boosting/statistics/quantization_no.hpp"

namespace boosting {

    class NoDenseDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<CContiguousView<Tuple<float64>>>> quantizationMatrixPtr_;

        public:

            NoDenseDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<CContiguousView<Tuple<float64>>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              std::optional<IQuantization::DenseDecomposableMatrixVisitor> denseDecomposableMatrixVisitor,
              std::optional<IQuantization::SparseDecomposableMatrixVisitor> sparseDecomposableMatrixVisitor,
              std::optional<IQuantization::DenseNonDecomposableMatrixVisitor> denseNonDecomposableMatrixVisitor)
              override {
                if (denseDecomposableMatrixVisitor) {
                    (*denseDecomposableMatrixVisitor)(quantizationMatrixPtr_);
                }
            }
    };

    class NoSparseDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<SparseSetView<Tuple<float64>>>> quantizationMatrixPtr_;

        public:

            NoSparseDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<SparseSetView<Tuple<float64>>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              std::optional<IQuantization::DenseDecomposableMatrixVisitor> denseDecomposableMatrixVisitor,
              std::optional<IQuantization::SparseDecomposableMatrixVisitor> sparseDecomposableMatrixVisitor,
              std::optional<IQuantization::DenseNonDecomposableMatrixVisitor> denseNonDecomposableMatrixVisitor)
              override {
                if (sparseDecomposableMatrixVisitor) {
                    (*sparseDecomposableMatrixVisitor)(quantizationMatrixPtr_);
                }
            }
    };

    class NoDenseNonDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView>> quantizationMatrixPtr_;

        public:

            NoDenseNonDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {}

            void quantize(const PartialIndexVector& outputIndices) override {}

            void visitQuantizationMatrix(
              std::optional<IQuantization::DenseDecomposableMatrixVisitor> denseDecomposableMatrixVisitor,
              std::optional<IQuantization::SparseDecomposableMatrixVisitor> sparseDecomposableMatrixVisitor,
              std::optional<IQuantization::DenseNonDecomposableMatrixVisitor> denseNonDecomposableMatrixVisitor)
              override {
                if (denseNonDecomposableMatrixVisitor) {
                    (*denseNonDecomposableMatrixVisitor)(quantizationMatrixPtr_);
                }
            }
    };

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

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Tuple<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Tuple<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView>>(statisticMatrix));
            }
    };

    class NoQuantizationFactory final : public IQuantizationFactory {
        public:

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Tuple<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Tuple<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView>>(statisticMatrix));
            }
    };

    std::unique_ptr<IQuantizationFactory> NoQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<NoQuantizationFactory>();
    }

}
