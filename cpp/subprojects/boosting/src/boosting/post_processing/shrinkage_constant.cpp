#include "boosting/post_processing/shrinkage_constant.hpp"
#include "common/validation.hpp"


namespace boosting {

    ConstantShrinkage::ConstantShrinkage(float64 shrinkage)
        : shrinkage_(shrinkage) {
        assertGreater(shrinkage, 0.0);
        assertSmaller(shrinkage, 1.0);
    }

    void ConstantShrinkage::postProcess(AbstractPrediction& prediction) const {
        uint32 numElements = prediction.getNumElements();
        AbstractPrediction::score_iterator iterator = prediction.scores_begin();

        for (uint32 i = 0; i < numElements; i++) {
            iterator[i] *= shrinkage_;
        }
    }

}
