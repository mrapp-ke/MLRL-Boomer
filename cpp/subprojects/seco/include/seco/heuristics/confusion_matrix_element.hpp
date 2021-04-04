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

}
