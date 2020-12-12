/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "feature_binning.h"


/**
 * Assigns feature values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinning final : public IFeatureBinning {

    private:

        float32 binRatio_;

    public:

        /**
         * @param binRatio A percentage that specifies how many bins should be used to assign the values in an array to,
         *                 e.g., if 100 values are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
         */
        EqualFrequencyFeatureBinning(float32 binRatio);

        FeatureInfo getFeatureInfo(FeatureVector& featureVector) const override;

        void createBins(FeatureInfo featureInfo, const FeatureVector& featureVector,
                        IBinningObserver<float32>& observer) const override;

};
