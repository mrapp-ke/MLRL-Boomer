/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "common/data/types.hpp"


namespace seco {

    /**
     * An enum that specifies all positive elements of a confusion matrix.
     */
    enum ConfusionMatrixElement : uint32 {
        IN = 0,
        IP = 1,
        RN = 2,
        RP = 3
    };

    /**
     * Returns the confusion matrix element, a label corresponds to, depending on the ground truth an a prediction.
     *
     * @param trueLabel         The true label according to the ground truth
     * @param majorityLabel     The prediction of the default rule. The prediction is assumed to be the inverse
     * @return                  The confusion matrix element
     */
    static inline ConfusionMatrixElement getConfusionMatrixElement(bool trueLabel, bool majorityLabel) {
        if (trueLabel) {
            return majorityLabel ? RN : RP;
        } else {
            return majorityLabel ? IN : IP;
        }
    }

}
