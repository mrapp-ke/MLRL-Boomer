/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "config.hpp"

#if SIMD_SUPPORT_ENABLED
    #include <xsimd/xsimd.hpp>
#endif

namespace util {

    /**
     * Returns the names of all instruction set extensions for SIMD operations available on the machine.
     *
     * @return An `std::vector` that contains the names of all supported instruction set extensions
     */
    static inline std::vector<std::string> getSupportedSimdExtensions() {
        std::vector<std::string> names;

#if SIMD_SUPPORT_ENABLED
        xsimd::all_architectures::for_each([&](auto architecture) {
            if (xsimd::available_architectures().has(architecture)) {
                names.emplace_back(architecture.name());
            }
        });
#endif

        return names;
    }
}
