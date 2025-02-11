#include "mlrl/boosting/post_processing/shrinkage_constant.hpp"

#include "mlrl/common/util/validation.hpp"

namespace boosting {

    template<typename ScoreIterator>
    static inline void postProcessInternally(ScoreIterator scoresBegin, ScoreIterator scoresEnd, float32 shrinkage) {
        uint32 numElements = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numElements; i++) {
            scoresBegin[i] *= shrinkage;
        }
    }

    /**
     * Post-processes the predictions of rules by shrinking their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkage final : public IPostProcessor {
        private:

            const float32 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1)
             */
            ConstantShrinkage(float32 shrinkage) : shrinkage_(shrinkage) {}

            /**
             * @see `IPostProcessor::postProcess`
             */
            void postProcess(View<float32>::iterator begin, View<float32>::iterator end) const override {
                postProcessInternally(begin, end, shrinkage_);
            }

            /**
             * @see `IPostProcessor::postProcess`
             */
            void postProcess(View<float64>::iterator begin, View<float64>::iterator end) const override {
                postProcessInternally(begin, end, shrinkage_);
            }
    };

    /**
     * Allows to create instances of the type `IPostProcessor` that post-process the predictions of rules by shrinking
     * their weights by a constant "shrinkage" parameter.
     */
    class ConstantShrinkageFactory final : public IPostProcessorFactory {
        private:

            const float32 shrinkage_;

        public:

            /**
             * @param shrinkage The value of the "shrinkage" parameter. Must be in (0, 1)
             */
            ConstantShrinkageFactory(float32 shrinkage) : shrinkage_(shrinkage) {}

            /**
             * @see `IPostProcessorFactory::create`
             */
            std::unique_ptr<IPostProcessor> create() const override {
                return std::make_unique<ConstantShrinkage>(shrinkage_);
            }
    };

    ConstantShrinkageConfig::ConstantShrinkageConfig() : shrinkage_(0.3f) {}

    float32 ConstantShrinkageConfig::getShrinkage() const {
        return shrinkage_;
    }

    IConstantShrinkageConfig& ConstantShrinkageConfig::setShrinkage(float32 shrinkage) {
        util::assertGreater<float32>("shrinkage", shrinkage, 0);
        util::assertLess<float32>("shrinkage", shrinkage, 1);
        shrinkage_ = shrinkage;
        return *this;
    }

    std::unique_ptr<IPostProcessorFactory> ConstantShrinkageConfig::createPostProcessorFactory() const {
        return std::make_unique<ConstantShrinkageFactory>(shrinkage_);
    }

}
