#include "boosting/post_processing/shrinkage_constant.hpp"
#include "common/util/validation.hpp"


namespace boosting {

    /**
     * Post-processes the predictions of rules by shrinking their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkage final : public IPostProcessor {

        private:

            float64 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1)
             */
            ConstantShrinkage(float64 shrinkage)
                : shrinkage_(shrinkage) {

            }

            void postProcess(AbstractPrediction& prediction) const override {
                uint32 numElements = prediction.getNumElements();
                AbstractPrediction::score_iterator iterator = prediction.scores_begin();

                for (uint32 i = 0; i < numElements; i++) {
                    iterator[i] *= shrinkage_;
                }
            }

    };

    float64 ConstantShrinkageConfig::getShrinkage() const {
        return shrinkage_;
    }

    ConstantShrinkageConfig& ConstantShrinkageConfig::setShrinkage(float64 shrinkage) {
        assertGreater<float64>("shrinkage", shrinkage, 0);
        assertLess<float64>("shrinkage", shrinkage, 1);
        shrinkage_ = shrinkage;
        return *this;
    }

    ConstantShrinkageFactory::ConstantShrinkageFactory(float64 shrinkage)
        : shrinkage_(shrinkage) {

    }

    std::unique_ptr<IPostProcessor> ConstantShrinkageFactory::create() const {
        return std::make_unique<ConstantShrinkage>(shrinkage_);
    }

}
