/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "config.hpp"

#include <string>
#include <vector>

#if SIMD_SUPPORT_ENABLED
    #include <stdexcept>
    #include <xsimd/xsimd.hpp>

    #define SIMD_ARCHITECTURE_auto 1
    #define SIMD_ARCHITECTURE_neon64 2
    #define SIMD_ARCHITECTURE_avx512f 3
    #define SIMD_ARCHITECTURE_avx2 4
    #define SIMD_ARCHITECTURE_avx 5
    #define SIMD_ARCHITECTURE_sse2 6

    #define SIMD_ARCHITECTURE_VALUE_IMPL(x) SIMD_ARCHITECTURE_##x
    #define SIMD_ARCHITECTURE_VALUE(x) SIMD_ARCHITECTURE_VALUE_IMPL(x)
#endif

namespace util {

#if SIMD_SUPPORT_ENABLED
    #if defined(__aarch64__) || defined(_M_ARM64)
        #if SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 1
    using simd_architectures = xsimd::arch_list<xsimd::neon64>;
        #elif SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 2
    using simd_architectures = xsimd::arch_list<xsimd::neon64>;
        #else
            #error "Invalid value given for macro SIMD_ARCHITECTURE"
        #endif
    #else
        #if SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 1
    using simd_architectures = xsimd::arch_list<xsimd::avx512f, xsimd::avx2, xsimd::avx, xsimd::sse2>;
        #elif SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 3
    using simd_architectures = xsimd::arch_list<xsimd::avx512f>;
        #elif SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 4
    using simd_architectures = xsimd::arch_list<xsimd::avx2>;
        #elif SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 5
    using simd_architectures = xsimd::arch_list<xsimd::avx>;
        #elif SIMD_ARCHITECTURE_VALUE(SIMD_ARCHITECTURE) == 6
    using simd_architectures = xsimd::arch_list<xsimd::sse2>;
        #else
            #error "Invalid value given for macro SIMD_ARCHITECTURE"
        #endif
    #endif
#endif

#if SIMD_SUPPORT_ENABLED
    /**
     * Loads an array into a SIMD register and returns the corresponding batch.
     *
     * @tparam batch    The type of the batch
     * @tparam T        The type of the array
     * @param array     The array to be loaded
     * @return          The batch
     */
    template<typename batch, typename T>
    static inline constexpr batch load_simd(T* array) {
    #if DEBUG
        if (!xsimd::is_aligned(array)) {
            throw std::runtime_error("Array to be loaded into SIMD register is not properly aligned");
        }
    #endif

        return batch::load_aligned(array);
    }
#endif

    /**
     * Returns the names of all instruction set extensions for SIMD operations available on the machine.
     *
     * @return An `std::vector` that contains the names of all supported instruction set extensions
     */
    static inline std::vector<std::string> getSupportedSimdExtensions() {
        std::vector<std::string> names;

#if SIMD_SUPPORT_ENABLED
        simd_architectures::for_each([&](auto architecture) {
            if (xsimd::available_architectures().has(architecture)) {
                names.emplace_back(architecture.name());
            }
        });
#endif

        return names;
    }
}
