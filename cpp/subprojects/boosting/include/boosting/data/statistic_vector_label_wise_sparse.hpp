/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


namespace boosting {

    /**
     * An one-dimensional sparse vector that stores gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function. For each element in the vector a single gradient and Hessian is stored.
     */
    class SparseLabelWiseStatisticVector final {

        public:

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             */
            SparseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            SparseLabelWiseStatisticVector(uint32 numElements, bool init);

    };

}
