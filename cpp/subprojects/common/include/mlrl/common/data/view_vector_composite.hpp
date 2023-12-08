/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/data/view_vector.hpp"

/**
 * A vector that is backed two one-dimensional views.
 *
 * @tparam FirstView    The type of the first view
 * @tparam SecondView   The type of the second view
 */
template<typename FirstView, typename SecondView>
class MLRLCOMMON_API CompositeVector : public CompositeView<FirstView, SecondView> {
    public:

        /**
         * @param firstView     The first view, the vector should be backed by
         * @param secondView    The second view, the vector should be backed by
         */
        CompositeVector(FirstView&& firstView, SecondView&& secondView)
            : CompositeView<FirstView, SecondView>(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `CompositeVector` that should be copied
         */
        CompositeVector(const CompositeVector<FirstView, SecondView>& other)
            : CompositeView<FirstView, SecondView>(other) {}

        /**
         * @param other A reference to an object of type `CompositeVector` that should be moved
         */
        CompositeVector(CompositeVector<FirstView, SecondView>&& other)
            : CompositeView<FirstView, SecondView>(std::move(other)) {}

        virtual ~CompositeVector() override {}

        /**
         * Sets all values stored in the view to zero.
         */
        void clear() {
            CompositeView<FirstView, SecondView>::firstView.clear();
            CompositeView<FirstView, SecondView>::secondView.clear();
        }
};
