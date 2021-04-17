/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * An implementation of the class `ILabelSubSampling` that does not perform any sampling, but includes all labels.
 */
class NoLabelSubSampling final : public ILabelSubSampling {

    private:

        uint32 numLabels_;

    public:

        /**
         * @param numLabels The total number of available labels
         */
        NoLabelSubSampling(uint32 numLabels);

        std::unique_ptr<IIndexVector> subSample(RNG& rng) const override;

};

/**
 * Allows to create objects of the class `ILabelSubSampling` that do not perform any sampling, but include all labels.
 */
class NoLabelSubSamplingFactory final : public ILabelSubSamplingFactory {

    public:

        std::unique_ptr<ILabelSubSampling> create(uint32 numLabels) const override;

};
