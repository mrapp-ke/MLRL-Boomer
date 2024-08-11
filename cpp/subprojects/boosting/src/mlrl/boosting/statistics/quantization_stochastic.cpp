#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename View>
    class StochasticQuantizationMatrix final : public IQuantizationMatrix<CContiguousView<Tuple<float64>>> {
        private:

            const View& view_;

            // TODO Use correct type
            MatrixDecorator<AllocatedCContiguousView<Tuple<float64>>> matrix_;

        public:

            StochasticQuantizationMatrix(const View& view)
                : view_(view), matrix_(AllocatedCContiguousView<Tuple<float64>>(view.numRows, view.numCols)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<CContiguousView<Tuple<float64>>>::view_type& getView() const override {
                return matrix_.getView();
            }
    };

    template<typename View>
    class StochasticQuantization final : public IQuantization {
        private:

            std::unique_ptr<IQuantizationMatrix<CContiguousView<Tuple<float64>>>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(const View& view)
                : quantizationMatrixPtr_(std::make_unique<StochasticQuantizationMatrix<View>>(view)) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor denseDecomposableMatrixVisitor,
              IQuantization::SparseDecomposableMatrixVisitor sparseDecomposableMatrixVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor denseNonDecomposableMatrixVisitor) override {
                denseDecomposableMatrixVisitor(quantizationMatrixPtr_);
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
                return std::make_unique<StochasticQuantization<CContiguousView<Tuple<float64>>>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(const SparseSetView<Tuple<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Tuple<float64>>>>(statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<DenseNonDecomposableStatisticView>>(statisticMatrix);
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
