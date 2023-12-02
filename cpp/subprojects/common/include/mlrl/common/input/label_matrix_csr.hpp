/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"

/**
 * Defines an interface for all label matrices that provide row-wise access to the labels of individual examples that
 * are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrLabelMatrix : virtual public IRowWiseLabelMatrix {
    public:

        virtual ~ICsrLabelMatrix() override {}
};

/**
 * Implements row-wise read-only access to the labels of individual training examples that are stored in a pre-allocated
 * sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrLabelMatrix final : public IterableBinarySparseMatrixDecorator<MatrixDecorator<BinaryCsrView>>,
                             public ICsrLabelMatrix {
    public:

        /**
         * @param indices A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the
         *                column-indices, the relevant labels correspond to
         * @param indptr  A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the
         *                first element in `indices` that corresponds to a certain row. The index at the last position
         *                is equal to `numNonZeroValues`
         * @param numRows The number of rows in the label matrix
         * @param numCols The number of columns in the label matrix
         */
        CsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols);

        bool isSparse() const override;

        uint32 getNumExamples() const override;

        uint32 getNumLabels() const override;

        float32 calculateLabelCardinality() const override;

        std::unique_ptr<LabelVector> createLabelVector(uint32 row) const override;

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const override;

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IPartitionSamplingFactory& factory) const override;

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  const SinglePartition& partition,
                                                                  IStatistics& statistics) const override;

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  BiPartition& partition,
                                                                  IStatistics& statistics) const override;

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override;
};

/**
 * Creates and returns a new object of the type `ICsrLabelMatrix`.
 *
 * @param indices A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the column-indices,
 *                the relevant labels correspond to
 * @param indptr  A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of the first
 *                element in `indices` that corresponds to a certain row. The index at the last position is equal to
 *                `numNonZeroValues`
 * @param numRows The number of rows in the label matrix
 * @param numCols The number of columns in the label matrix
 * @return        An unique pointer to an object of type `ICsrLabelMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrLabelMatrix> createCsrLabelMatrix(uint32* indices, uint32* indptr, uint32 numRows,
                                                                     uint32 numCols);

#ifdef _WIN32
    #pragma warning(pop)
#endif
