#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename View, typename StatisticType>
    class StochasticQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float32>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseDecomposable32BitVisitor) {
                    (*denseDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<CContiguousView<Statistic<float64>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseDecomposable64BitVisitor) {
                    (*denseDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float32>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (sparseDecomposable32BitVisitor) {
                    (*sparseDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float64>>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (sparseDecomposable64BitVisitor) {
                    (*sparseDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable32BitVisitor) {
                    (*denseNonDecomposable32BitVisitor)(quantizationMatrix);
                }
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>& quantizationMatrix,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable64BitVisitor) {
                    (*denseNonDecomposable64BitVisitor)(quantizationMatrix);
                }
            }

            std::unique_ptr<IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(
              std::unique_ptr<IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) override {
                visitInternally(*quantizationMatrixPtr_, denseDecomposable32BitVisitor, denseDecomposable64BitVisitor,
                                sparseDecomposable32BitVisitor, sparseDecomposable64BitVisitor,
                                denseNonDecomposable32BitVisitor, denseNonDecomposable64BitVisitor);
            }
    };

    template<typename View, typename StatisticType>
    class StochasticQuantizationMatrix final : public IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>> {
        private:

            const View& view_;

            // TODO Use correct type
            MatrixDecorator<AllocatedCContiguousView<Statistic<StatisticType>>> matrix_;

        public:

            StochasticQuantizationMatrix(const View& view)
                : view_(view), matrix_(AllocatedCContiguousView<Statistic<StatisticType>>(view.numRows, view.numCols)) {
            }

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<CContiguousView<Statistic<StatisticType>>>::view_type& getView()
              const override {
                return matrix_.getView();
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Statistic<float32>>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Statistic<float32>>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Statistic<float64>>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Statistic<float64>>, float64>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float32>>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Statistic<float32>>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float64>>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Statistic<float64>>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix));
            }
    };

    class StochasticQuantizationFactory final : public IQuantizationFactory {
        private:

            uint32 numBits_;

        public:

            /**
             * @param numBits The number of bits to be used for quantized statistics
             */
            StochasticQuantizationFactory(uint32 numBits) : numBits_(numBits) {}

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Statistic<float32>>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Statistic<float32>>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Statistic<float64>>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Statistic<float64>>, float64>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float32>>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Statistic<float32>>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float64>>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Statistic<float64>>, float64>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix));
            }
    };

    StochasticQuantizationConfig::StochasticQuantizationConfig() : numBits_(4) {}

    uint32 StochasticQuantizationConfig::getNumBits() const {
        return numBits_;
    }

    IStochasticQuantizationConfig& StochasticQuantizationConfig::setNumBits(uint32 numBits) {
        util::assertGreater<uint32>("numBits", numBits, 0);
        numBits_ = numBits;
        return *this;
    }

    std::unique_ptr<IQuantizationFactory> StochasticQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<StochasticQuantizationFactory>(numBits_);
    }

}
