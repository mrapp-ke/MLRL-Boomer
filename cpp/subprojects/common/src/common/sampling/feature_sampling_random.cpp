#include "common/sampling/feature_sampling_random.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/indices/index_iterator.hpp"
#include "index_sampling.hpp"
#include <cmath>


/**
 * Allows to select a subset of the available features without replacement.
 */
class RandomFeatureSubsetSelection final : public IFeatureSubSampling {

    private:

        uint32 numFeatures_;

        PartialIndexVector indexVector_;

    public:

        /**
         * @param numFeatures   The total number of available features
         * @param sampleSize    The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available features). Must be in (0, 1) or 0, if the default sample size
         *                      `floor(log2(num_features - 1) + 1)` should be used
         */
        RandomFeatureSubsetSelection(uint32 numFeatures, float32 sampleSize)
            : numFeatures_(numFeatures),
              indexVector_(PartialIndexVector((uint32) (sampleSize > 0 ? sampleSize * numFeatures
                                                                       : log2(numFeatures - 1) + 1))) {

        }

        const IIndexVector& subSample(RNG& rng) override {
            sampleIndicesWithoutReplacement<IndexIterator>(indexVector_, IndexIterator(numFeatures_), numFeatures_,
                                                           rng);
            return indexVector_;
        }

};

RandomFeatureSubsetSelectionFactory::RandomFeatureSubsetSelectionFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IFeatureSubSampling> RandomFeatureSubsetSelectionFactory::create(uint32 numFeatures) const {
    return std::make_unique<RandomFeatureSubsetSelection>(numFeatures, sampleSize_);
}
