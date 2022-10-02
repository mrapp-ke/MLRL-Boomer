/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/util/quality.hpp"


namespace seco {

    /**
     * Returns whether the quality of a rule is better than the quality of a second one.
     *
     * @param first     An object of type `Quality` that represents the quality of the first rule
     * @param second    An object of type `Quality` that represents the quality of the second rule
     * @return          True, if the quality of the first rule is better than the quality of the second one, false
     *                  otherwise
     */
    static inline constexpr bool compareSeCoRuleQuality(const Quality& first, const Quality& second) {
        return first.quality < second.quality;
    }

}
