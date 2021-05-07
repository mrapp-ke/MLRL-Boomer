#pragma once

#include "common/sampling/instance_sampling.hpp"
#include "boosting/statistics/statistics_boosting.hpp"
#include "boosting/data/matrix_dense_example_wise.hpp"
#include "boosting/data/matrix_dense_label_wise.hpp"
#include <algorithm>

typedef std::vector<uint32> uint32Vector;
typedef std::vector<float32> float32Vector;

namespace boosting{
    class GradientBasedLabelSet: public IInstanceSubSampling,
                                 public IVisitor{
        private:
            float32 sampleSizeTop_;
            float32 sampleSizeRandom_;

        public:
            GradientBasedLabelSet(float32 sampleSizeTop, float32 sampleSizeRandom);

            std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng,
                                                     const IRandomAccessLabelMatrix& labelMatrix,
                                                     const IStatistics& statistics)const override;

            std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng,
                                                     const IRandomAccessLabelMatrix& labelMatrix,
                                                     const IStatistics& statistics) const override;

            std::unique_ptr<IWeightVector> subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                      RNG& rng, const IStatistics& statistics)const;
            template<typename DenseStatisticMatrix>
            const mapExamplesGradients findExamplesPerLabelset(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                const DenseStatisticMatrix& statisticMatrix)const;

            uint32Vector getNextLabelset(mapExamplesGradients examplesPerLabelset)const;

            uint32 getNextSubset(float32Vector& desiredSamplesPerLabelset, uint32Vector& desiredSamplesPerSet)const;

            const mapExamplesGradients visit(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                             const DenseExampleWiseStatisticMatrix& statisticMatrix)const;

            const mapExamplesGradients visit(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                             const DenseLabelWiseStatisticMatrix& statisticMatrix)const;
    };
}