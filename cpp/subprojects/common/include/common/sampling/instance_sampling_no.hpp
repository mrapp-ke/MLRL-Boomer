/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSubSampling` that do not perform any sampling, but assign equal
 * weights to all examples.
 */
class NoInstanceSubSamplingFactory final : public IInstanceSubSamplingFactory {

    public:

        std::unique_ptr<IInstanceSubSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IInstanceSubSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
