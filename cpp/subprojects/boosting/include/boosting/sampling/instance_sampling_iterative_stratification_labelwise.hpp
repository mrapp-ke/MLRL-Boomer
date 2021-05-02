#pragma once
#include "common/data/types.hpp"
#include "common/sampling/instance_sampling.hpp"
#include <vector>
#include <map>
#include <list>

typedef std::map<std::vector<uint32>, std::list<uint32>> mapExamples;
typedef std::vector<uint32> uint32Vector;
typedef std::vector<float32> float32Vector;

namespace boosting{
    class IterativeStratificationLabelWise final : public IInstanceSubSampling {
        private:
            float32 sampleSize_;

        public:
            IterativeStratificationLabelWise(float32 sampleSize);

            std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng,
                                                     const IRandomAccessLabelMatrix& labelMatrix,
                                                     const IStatistics& statistics)const override;

            std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng,
                                                     const IRandomAccessLabelMatrix& labelMatrix,
                                                     const IStatistics& statistics) const override;

            mapExamples findExamplesPerLabel(const IRandomAccessLabelMatrix& labelMatrixPtr)const;

            uint32Vector getNextLabel(mapExamples& examplesPerLabel)const;

            uint32 getNextSubset(float32Vector& desiredSamplesPerLabel,
                                 uint32Vector& desiredSamplesPerSet)const;

            std::unique_ptr<IWeightVector> subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr, RNG& rng)const;
       };
}