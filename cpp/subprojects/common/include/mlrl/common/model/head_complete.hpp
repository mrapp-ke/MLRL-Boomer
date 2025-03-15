/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/model/head.hpp"

#include <type_traits>

/**
 * A head that contains a numerical score for each available output.
 *
 * @tparam ScoreType The type of the numerical scores
 */
template<typename ScoreType>
class MLRLCOMMON_API CompleteHead final : public VectorDecorator<AllocatedVector<ScoreType>>,
                                          public IHead {
    public:

        /**
         * @param numElements The number of scores that are contained by the head.
         */
        CompleteHead(uint32 numElements)
            : VectorDecorator<AllocatedVector<ScoreType>>(AllocatedVector<ScoreType>(numElements)) {}

        /**
         * An iterator that provides access to the scores the are contained by the head and allows to modify them.
         */
        typedef typename View<ScoreType>::iterator value_iterator;

        /**
         * An iterator that provides read-only access to the scores that are contained by the head.
         */
        typedef typename View<ScoreType>::const_iterator value_const_iterator;

        /**
         * Returns a `value_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin() {
            return this->view.begin();
        }

        /**
         * Returns a `value_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end() {
            return this->view.end();
        }

        /**
         * Returns a `value_const_iterator` to the beginning of the scores that are contained by the head.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const {
            return this->view.cbegin();
        }

        /**
         * Returns a `value_const_iterator` to the end of the scores that are contained by the head.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const {
            return this->view.cend();
        }

        void visit(CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                   CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                   PartialHeadVisitor<float32> partial32BitHeadVisitor,
                   PartialHeadVisitor<float64> partial64BitHeadVisitor) const override {
            if constexpr (std::is_same_v<ScoreType, float32>) {
                complete32BitHeadVisitor(*this);
            } else if constexpr (std::is_same_v<ScoreType, float64>) {
                complete64BitHeadVisitor(*this);
            } else {
                throw std::runtime_error("No visitor available for handling object of template class CompleteHead");
            }
        }
};
