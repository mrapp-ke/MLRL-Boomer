#include "mlrl/boosting/statistics/quantization_stochastic.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"
#include "mlrl/common/util/validation.hpp"
#include "statistics_decomposable_dense.hpp"

namespace boosting {

    template<typename View, typename StatisticType, typename VectorMath>
    class StochasticQuantizationMatrix final
        : public IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>> {
        private:

            const View& view_;

            // TODO Use correct type
            DenseDecomposableStatisticMatrix<StatisticType, VectorMath> matrix_;

        public:

            StochasticQuantizationMatrix(const View& view) : view_(view), matrix_(view.numRows, view.numCols) {}

            void quantize(const CompleteIndexVector& outputIndices) override {
                // TODO Implement
            }

            void quantize(const PartialIndexVector& outputIndices) override {
                // TODO Implement
            }

            const typename IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>>::view_type& getView()
              const override {
                return matrix_.getView();
            }
    };

    template<typename View, typename StatisticType, typename VectorMath>
    class StochasticQuantization final : public IQuantization {
        private:

            static inline void visitInternally(
              const IQuantizationMatrix<DenseDecomposableStatisticView<float32>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseDecomposableStatisticView<float64>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseDecomposable64BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float32>>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                sparseDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<SparseSetView<Statistic<float64>>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                sparseDecomposable64BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float32>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseNonDecomposable32BitVisitor(quantizationMatrix);
            }

            static inline void visitInternally(
              const IQuantizationMatrix<DenseNonDecomposableStatisticView<float64>>& quantizationMatrix,
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) {
                denseNonDecomposable64BitVisitor(quantizationMatrix);
            }

            std::unique_ptr<IQuantizationMatrix<DenseDecomposableStatisticView<StatisticType>>> quantizationMatrixPtr_;

        public:

            StochasticQuantization(const View& view)
                : quantizationMatrixPtr_(
                    std::make_unique<StochasticQuantizationMatrix<View, StatisticType, VectorMath>>(view)) {}

            void visitQuantizationMatrix(
              IQuantization::DenseDecomposableMatrixVisitor<float32> denseDecomposable32BitVisitor,
              IQuantization::DenseDecomposableMatrixVisitor<float64> denseDecomposable64BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float32> sparseDecomposable32BitVisitor,
              IQuantization::SparseDecomposableMatrixVisitor<float64> sparseDecomposable64BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float32> denseNonDecomposable32BitVisitor,
              IQuantization::DenseNonDecomposableMatrixVisitor<float64> denseNonDecomposable64BitVisitor) override {
                visitInternally(*quantizationMatrixPtr_, denseDecomposable32BitVisitor, denseDecomposable64BitVisitor,
                                sparseDecomposable32BitVisitor, sparseDecomposable64BitVisitor,
                                denseNonDecomposable32BitVisitor, denseNonDecomposable64BitVisitor);
            }
    };

    /**
     * Allows to to create instances of the type `IQuantization` that uses a stochastic rounding strategy.
     */
    template<typename VectorMath>
    class StochasticQuantizationFactory final : public IQuantizationFactory {
        private:

            uint8 numBins_;

        public:

            /**
             * @param numBins The number of bins to be used for quantized statistics
             */
            StochasticQuantizationFactory(uint8 numBins) : numBins_(numBins) {}

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<
                  StochasticQuantization<DenseDecomposableStatisticView<float32>, float32, VectorMath>>(
                  statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<
                  StochasticQuantization<DenseDecomposableStatisticView<float64>, float64, VectorMath>>(
                  statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float32>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float32>>, float32, VectorMath>>(
                  statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const SparseSetView<Statistic<float64>>& statisticMatrix) const override {
                return std::make_unique<StochasticQuantization<SparseSetView<Statistic<float64>>, float64, VectorMath>>(
                  statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float32>& statisticMatrix) const override {
                return std::make_unique<
                  StochasticQuantization<DenseNonDecomposableStatisticView<float32>, float32, VectorMath>>(
                  statisticMatrix);
            }

            std::unique_ptr<IQuantization> create(
              const DenseNonDecomposableStatisticView<float64>& statisticMatrix) const override {
                return std::make_unique<
                  StochasticQuantization<DenseNonDecomposableStatisticView<float64>, float64, VectorMath>>(
                  statisticMatrix);
            }
    };

    StochasticQuantizationConfig::StochasticQuantizationConfig(ReadableProperty<ISimdConfig> simdConfig)
        : simdConfig_(simdConfig), numBins_(16) {}

    uint8 StochasticQuantizationConfig::getNumBins() const {
        return numBins_;
    }

    IStochasticQuantizationConfig& StochasticQuantizationConfig::setNumBins(uint8 numBins) {
        util::assertGreater<uint8>("numBins", numBins, 0);
        numBins_ = numBins;
        return *this;
    }

    std::unique_ptr<IQuantizationFactory> StochasticQuantizationConfig::createQuantizationFactory(
      const IOutputMatrix& outputMatrix) const {
#if SIMD_SUPPORT_ENABLED
        if (simdConfig_.get().isSimdRecommended(outputMatrix.getNumOutputs())) {
            return std::make_unique<StochasticQuantizationFactory<SimdVectorMath>>(numBins_);
        }
#endif

        return std::make_unique<StochasticQuantizationFactory<SequentialVectorMath>>(numBins_);
    }

}
