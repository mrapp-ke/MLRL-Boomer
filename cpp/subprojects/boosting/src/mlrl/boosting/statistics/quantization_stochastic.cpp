#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/data/matrix_bit_integer.hpp"
#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename View>
    class StochasticQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<BitMatrix>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(std::unique_ptr<IQuantizationMatrix<BitMatrix>> quantizationMatrixPtr)
                : quantizationMatrixPtr_(std::move(quantizationMatrixPtr)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            void visitQuantizationMatrix(
              std::optional<IQuantization::DenseDecomposableMatrixVisitor> denseDecomposableMatrixVisitor,
              std::optional<IQuantization::SparseDecomposableMatrixVisitor> sparseDecomposableMatrixVisitor,
              std::optional<IQuantization::DenseNonDecomposableMatrixVisitor> denseNonDecomposableMatrixVisitor)
              override {
                // TODO Implement
            }
    };

    template<typename View>
    class StochasticQuantizationMatrix final : public IQuantizationMatrix<BitMatrix> {
        private:

            const View& view_;

            IntegerBitMatrix matrix_;

        public:

            StochasticQuantizationMatrix(const View& view, uint32 numBits)
                : view_(view), matrix_(view.numRows, view.numCols, numBits, true) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<BitMatrix>::view_type& getView() const override {
                return matrix_.getView();
            }

            std::unique_ptr<IQuantization> create(
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Tuple<float64>>>>(
                    statisticMatrix, matrix_.getView().numBitsPerElement));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Tuple<float64>>>>(
                    statisticMatrix, matrix_.getView().numBitsPerElement));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView>>(
                    statisticMatrix, matrix_.getView().numBitsPerElement));
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
              const CContiguousView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<CContiguousView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<CContiguousView<Tuple<float64>>>>(statisticMatrix,
                                                                                                  numBits_));
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Tuple<float64>>>>(
                  std::make_unique<StochasticQuantizationMatrix<SparseSetView<Tuple<float64>>>>(statisticMatrix,
                                                                                                numBits_));
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView>>(
                  std::make_unique<StochasticQuantizationMatrix<DenseNonDecomposableStatisticView>>(statisticMatrix,
                                                                                                    numBits_));
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
