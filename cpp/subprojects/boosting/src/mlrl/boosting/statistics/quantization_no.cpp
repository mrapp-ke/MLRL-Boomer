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

    template<typename StatisticType>
    class NoDenseDecomposableQuantization final : public IQuantization {
        private:

            NoQuantizationMatrix<CContiguousView<Statistic<StatisticType>>> quantizationMatrix_;

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float32>>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor) {
                denseDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float64>>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor) {
                denseDecomposable64BitVisitor(quantizationMatrix);
            }

        public:

            NoDenseDecomposableQuantization(const CContiguousView<Statistic<StatisticType>>& view)
                : quantizationMatrix_(view) {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) override {
                visitInternally(quantizationMatrix_, denseDecomposable32BitVisitor, denseDecomposable64BitVisitor);
            }
    };

    template<typename StatisticType>
    class NoSparseDecomposableQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float32>>>& quantizationMatrix,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor) {
                sparseDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float64>>>& quantizationMatrix,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor) {
                sparseDecomposable64BitVisitor(quantizationMatrix);
            }

            NoQuantizationMatrix<SparseSetView<Statistic<StatisticType>>> quantizationMatrix_;

        public:

            NoSparseDecomposableQuantization(const SparseSetView<Statistic<StatisticType>>& view)
                : quantizationMatrix_(view) {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) override {
                visitInternally(quantizationMatrix_, sparseDecomposable32BitVisitor, sparseDecomposable64BitVisitor);
            }
    };

    template<typename StatisticType>
    class NoDenseNonDecomposableQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>& quantizationMatrix,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseNonDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>& quantizationMatrix,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseNonDecomposable64BitVisitor(quantizationMatrix);
            }

            NoQuantizationMatrix<DenseNonDecomposableStatisticView<StatisticType>> quantizationMatrix_;

        public:

            NoDenseNonDecomposableQuantization(const DenseNonDecomposableStatisticView<StatisticType>& view)
                : quantizationMatrix_(view) {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) override {
                visitInternally(quantizationMatrix_, denseNonDecomposable32BitVisitor,
                                denseNonDecomposable64BitVisitor);
            }
    };

    class NoQuantizationFactory final : public IQuantizationFactory {
        public:

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float32>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float64>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float32>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float64>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float32>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float64>>(statisticMatrix);
            }
    };

    std::unique_ptr<IQuantizationFactory> NoQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<NoQuantizationFactory>();
    }

}
