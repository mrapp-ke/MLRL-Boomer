/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure which data type should be used for representing
     * statistics about the quality of predictions for training examples.
     */
    class IStatisticTypeConfig {
        public:

            virtual ~IStatisticTypeConfig() {}
    };
}
