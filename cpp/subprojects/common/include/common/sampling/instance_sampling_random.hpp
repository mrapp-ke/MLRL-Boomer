/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSubSampling` that allow to select a subset of the available training
 * examples without replacement.
 */
class RandomInstanceSubsetSelectionFactory final : public IInstanceSubSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        RandomInstanceSubsetSelectionFactory(float32 sampleSize);

        std::unique_ptr<IInstanceSubSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IInstanceSubSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
