#include "mlrl/boosting/statistics/quantization_no.hpp"

namespace boosting {

    template<typename StatisticType>
    class NoDenseDecomposableQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>> quantizationMatrixPtr_;

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float32>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor) {
                if (denseDecomposable32BitVisitor) {
                    (*denseDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float64>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor) {
                if (denseDecomposable64BitVisitor) {
                    (*denseDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

        public:

            NoDenseDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) override {
                visitInternally(*quantizationMatrixPtr_, denseDecomposable32BitVisitor, denseDecomposable64BitVisitor);
            }
    };

    template<typename StatisticType>
    class NoSparseDecomposableQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float32>>>& quantizationMatrix,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor) {
                if (sparseDecomposable32BitVisitor) {
                    (*sparseDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float64>>>& quantizationMatrix,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor) {
                if (sparseDecomposable64BitVisitor) {
                    (*sparseDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

            std::unique_ptr<IQuantizationMatrix<SparseSetView<Statistic<StatisticType>>>> quantizationMatrixPtr_;

        public:

            NoSparseDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<SparseSetView<Statistic<StatisticType>>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) override {
                visitInternally(*quantizationMatrixPtr_, sparseDecomposable32BitVisitor,
                                sparseDecomposable64BitVisitor);
            }
    };

    template<typename StatisticType>
    class NoDenseNonDecomposableQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>& quantizationMatrix,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable32BitVisitor) {
                    (*denseNonDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>& quantizationMatrix,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable64BitVisitor) {
                    (*denseNonDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

            std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView<StatisticType>>>
              quantizationMatrixPtr_;

        public:

            NoDenseNonDecomposableQuantization(
              std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView<StatisticType>>>
                quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) override {
                visitInternally(*quantizationMatrixPtr_, denseNonDecomposable32BitVisitor,
                                denseNonDecomposable64BitVisitor);
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
              const CContiguousView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Statistic<float32>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Statistic<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Statistic<float32>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Statistic<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>>(statisticMatrix));
            }
    };

    class NoQuantizationFactory final : public IQuantizationFactory {
        public:

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Statistic<float32>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoDenseDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<CContiguousView<Statistic<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Statistic<float32>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<NoSparseDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<SparseSetView<Statistic<float64>>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float32>>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>>(statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<NoDenseNonDecomposableQuantization<float64>>(
                  std::make_unique<NoQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>>(statisticMatrix));
            }
    };

    std::unique_ptr<IQuantizationFactory> NoQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<NoQuantizationFactory>();
    }

}
