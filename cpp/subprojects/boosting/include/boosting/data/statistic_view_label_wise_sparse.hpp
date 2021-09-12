/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once


namespace boosting {

    /**
     * Implements row-wise read-only access to the gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticConstView {

    };

    /**
     * Implements row-wise read and write access to the gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function and are stored in a pre-allocated matrix in the list of lists (LIL) format.
     */
    class SparseLabelWiseStatisticView : public SparseLabelWiseStatisticConstView {

    };

}
