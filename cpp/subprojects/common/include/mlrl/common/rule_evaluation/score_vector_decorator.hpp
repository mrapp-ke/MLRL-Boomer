/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/rule_evaluation/score_vector.hpp"

/**
 * An abstract base class for all vectors that store scores that may be predicted by a rule and are backed by a view.
 *
 * @tparam View The type of the view, the vector is backed by
 */
template<typename View>
class AbstractScoreVectorViewDecorator : public ViewDecorator<View>,
                                         virtual public IScoreVector {
    public:

        /**
         * @param view The view, the vector should be backed by
         */
        explicit AbstractScoreVectorViewDecorator(View&& view) : ViewDecorator<View>(std::move(view)) {}

        virtual ~AbstractScoreVectorViewDecorator() override {}

        /**
         * Returns the number of outputs for which the rule may predict.
         *
         * @return The number of outputs
         */
        uint32 getNumElements() const {
            return this->view.getNumElements();
        }

        /**
         * Returns whether the rule may only predict for a subset of the available outputs, or not.
         *
         * @return True, if the rule may only predict for a subset of the available outputs, false otherwise
         */
        bool isPartial() const {
            return this->view.isPartial();
        }

        /**
         * Sets the quality of the rule.
         *
         * @param quality The quality to be set
         */
        void setQuality(float64 quality) {
            this->view.quality = quality;
        }

        float64 getQuality() const override {
            return this->view.quality;
        }
};
