#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/boosting/data/view_statistic_decomposable_bit.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename View>
    class StochasticQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<BitDecomposableStatisticView>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(
              std::unique_ptr<IQuantizationMatrix<BitDecomposableStatisticView>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void visitQuantizationMatrix(
              std::optional<IQuantization::DenseDecomposableMatrixVisitor> denseDecomposableMatrixVisitor,
              std::optional<BitDecomposableMatrixVisitor> bitDecomposableMatrixVisitor,
              std::optional<IQuantization::SparseDecomposableMatrixVisitor> sparseDecomposableMatrixVisitor,
              std::optional<IQuantization::DenseNonDecomposableMatrixVisitor> denseNonDecomposableMatrixVisitor)
              override {
                if (bitDecomposableMatrixVisitor) {
                    (*bitDecomposableMatrixVisitor)(quantizationMatrixPtr_);
                }
            }
    };

    template<typename View>
    class StochasticQuantizationMatrix final : public IQuantizationMatrix<BitDecomposableStatisticView> {
        private:

            std::shared_ptr<RNG> rngPtr_;

            const View& view_;

            BitDecomposableStatisticView quantizedView_;

        public:

            StochasticQuantizationMatrix(std::shared_ptr<RNG> rngPtr, const View& view, uint32 numBits)
                : rngPtr_(std::move(rngPtr)), view_(view), quantizedView_(view.numRows, view.numCols, numBits) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<BitDecomposableStatisticView>::view_type& getView() const override {
                return quantizedView_;
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Tuple<float64>>>>(
                    rngPtr_, statisticMatrix, quantizedView_.firstView.numBitsPerElement));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Tuple<float64>>>>(
                    rngPtr_, statisticMatrix, quantizedView_.firstView.numBitsPerElement));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView>>(
                    rngPtr_, statisticMatrix, quantizedView_.firstView.numBitsPerElement));
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
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Tuple<float64>>>>(
                    rngFactoryPtr_->create(), statisticMatrix, numBits_));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Tuple<float64>>>>(
                    rngFactoryPtr_->create(), statisticMatrix, numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView>>(
                    rngFactoryPtr_->create(), statisticMatrix, numBits_));
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
