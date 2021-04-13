/*
 * @author Anna Kulischkin (Anna_Kulischkin@web.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSubSampling` that implement iterative stratified sampling as
 * proposed in the following publication:
 *
 * Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-label Data. In: Machine Learning and
 * Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer.
 */
class LabelWiseStratifiedSamplingFactory final : public IInstanceSubSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        LabelWiseStratifiedSamplingFactory(float32 sampleSize);

        std::unique_ptr<IInstanceSubSampling> create(const CContiguousLabelMatrix& labelMatrix) const override;

        std::unique_ptr<IInstanceSubSampling> create(const CsrLabelMatrix& labelMatrix) const override;

};
