/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/label_sampling.hpp"


/**
 * Allows to create objects of the class `ILabelSubSampling` that do not perform any sampling, but include all labels.
 */
class NoLabelSubSamplingFactory final : public ILabelSubSamplingFactory {

    public:

        std::unique_ptr<ILabelSubSampling> create(uint32 numLabels) const override;

};
