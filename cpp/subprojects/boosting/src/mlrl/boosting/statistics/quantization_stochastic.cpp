#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename View, typename StatisticType>
    class StochasticQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<DenseDecomposableStatisticView<float32>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseDecomposable32BitVisitor) {
                    (*denseDecomposable32BitVisitor)(quantizationMatrixPtr);
                }
            }

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<DenseDecomposableStatisticView<float64>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseDecomposable64BitVisitor) {
                    (*denseDecomposable64BitVisitor)(quantizationMatrixPtr);
                }
            }

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<SparseDecomposableStatisticView<float32>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (sparseDecomposable32BitVisitor) {
                    (*sparseDecomposable32BitVisitor)(quantizationMatrixPtr);
                }
            }

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<SparseDecomposableStatisticView<float64>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (sparseDecomposable64BitVisitor) {
                    (*sparseDecomposable64BitVisitor)(quantizationMatrixPtr);
                }
            }

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable32BitVisitor) {
                    (*denseNonDecomposable32BitVisitor)(quantizationMatrixPtr);
                }
            }

            static inline void visitInternally(
              std::unique_ptr<IQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>>& quantizationMatrixPtr,
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) {
                if (denseNonDecomposable64BitVisitor) {
                    (*denseNonDecomposable64BitVisitor)(quantizationMatrixPtr);
                }
            }

            std::unique_ptr<IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(
              std::unique_ptr<IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<DenseDecomposableMatrixVisitor<float32>> denseDecomposable32BitVisitor,
              std::optional<DenseDecomposableMatrixVisitor<float64>> denseDecomposable64BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float32>> sparseDecomposable32BitVisitor,
              std::optional<SparseDecomposableMatrixVisitor<float64>> sparseDecomposable64BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float32>> denseNonDecomposable32BitVisitor,
              std::optional<DenseNonDecomposableMatrixVisitor<float64>> denseNonDecomposable64BitVisitor) override {
                visitInternally(quantizationMatrixPtr_, denseDecomposable32BitVisitor, denseDecomposable64BitVisitor,
                                sparseDecomposable32BitVisitor, sparseDecomposable64BitVisitor,
                                denseNonDecomposable32BitVisitor, denseNonDecomposable64BitVisitor);
            }
    };

    template<typename View, typename StatisticType>
    class StochasticQuantizationMatrix final
        : public IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>> {
        private:

            const View& view_;

            std::shared_ptr<RNG> rngPtr_;

            uint32 numBits_;

            // TODO Use correct type
            MatrixDecorator<AllocatedCContiguousView<Statistic<StatisticType>>> matrix_;

        public:

            StochasticQuantizationMatrix(const View& view, std::shared_ptr<RNG> rngPtr, uint32 numBits)
                : view_(view), rngPtr_(std::move(rngPtr)), numBits_(numBits),
                  matrix_(AllocatedCContiguousView<Statistic<StatisticType>>(view.numRows, view.numCols)) {}

            void quantize(CompleteIndexVector::const_iterator outputIndicesBegin,
                          CompleteIndexVector::const_iterator outputIndicesEnd) override {
                // TODO Implement
            }

            void quantize(PartialIndexVector::const_iterator outputIndicesBegin,
                          PartialIndexVector::const_iterator outputIndicesEnd) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>>::view_type& getView()
              const override {
                return matrix_.getView();
            }

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngPtr_, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix, rngPtr_, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const SparseDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngPtr_, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const SparseDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseDecomposableStatisticView<float64>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseDecomposableStatisticView<float64>, float32>>(
                    statisticMatrix, rngPtr_, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngPtr_, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix, rngPtr_, numBits_));
            }
    };

    class StochasticQuantizationFactory final : public IQuantizationFactory {
        private:

            const std::unique_ptr<RNGFactory> rngFactoryPtr_;

            uint32 numBits_;

        public:

            StochasticQuantizationFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, uint32 numBits)
                : rngFactoryPtr_(std::move(rngFactoryPtr)), numBits_(numBits) {}

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const SparseDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const SparseDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float32>, float32>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float32>, float32>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView<float64>, float64>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView<float64>, float64>>(
                    statisticMatrix, rngFactoryPtr_->create(), numBits_));
            }
    };

    StochasticQuantizationConfig::StochasticQuantizationConfig(ReadableProperty<RNGConfig> rngConfig)
        : rngConfig_(rngConfig), numBits_(4) {}

    uint32 StochasticQuantizationConfig::getNumBits() const {
        return numBits_;
    }

    IStochasticQuantizationConfig& StochasticQuantizationConfig::setNumBits(uint32 numBits) {
        util::assertGreater<uint32>("numBits", numBits, 0);
        numBits_ = numBits;
        return *this;
    }

    std::unique_ptr<IQuantizationFactory> StochasticQuantizationConfig::createQuantizationFactory() const {
        return std::make_unique<StochasticQuantizationFactory>(rngConfig_.get().createRNGFactory(), numBits_);
    }

}
