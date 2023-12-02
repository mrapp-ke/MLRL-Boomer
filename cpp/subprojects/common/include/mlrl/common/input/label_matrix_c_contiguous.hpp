/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/input/label_matrix_row_wise.hpp"

/**
 * Defines an interface for all label matrices that provide row-wise access to the labels of individual examples that
 * are stored in a C-contiguous array.
 */
class MLRLCOMMON_API ICContiguousLabelMatrix : public IRowWiseLabelMatrix {
    public:

        virtual ~ICContiguousLabelMatrix() override {}
};

/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public IterableDenseMatrixDecorator<MatrixDecorator<CContiguousView<const uint8>>>,
                                     public ICContiguousLabelMatrix {
    public:

        /**
         * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         */
        CContiguousLabelMatrix(const uint8* array, uint32 numRows, uint32 numCols);

        /**
         * Provides read-only access to an individual row in the label matrix.
         */
        typedef const Vector<const uint8> const_row;

        /**
         * Creates and returns a view that provides read-only access to a specific row in the label matrix.
         *
         * @param row   The index of the row
         * @return      An object of type `const_row` that has been created
         */
        const_row operator[](uint32 row) const;

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
 * Creates and returns a new object of the type `ICContiguousLabelMatrix`.

 * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
 * @param numRows   The number of rows in the label matrix
 * @param numCols   The number of columns in the label matrix
 * @return          An unique pointer to an object of type `ICContiguousLabelMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICContiguousLabelMatrix> createCContiguousLabelMatrix(const uint8* array, uint32 numRows,
                                                                                     uint32 numCols);

#ifdef _WIN32
    #pragma warning(pop)
#endif
