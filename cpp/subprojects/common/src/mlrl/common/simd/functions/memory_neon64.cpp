#include "mlrl/common/simd/functions/memory_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template uint32 getPadding<xsimd::neon64, float32>(xsimd::neon64, uint32);
    template uint32 getPadding<xsimd::neon64, float64>(xsimd::neon64, uint32);
    template uint32 getPadding<xsimd::neon64, uint16>(xsimd::neon64, uint32);
    template uint32 getPadding<xsimd::neon64, uint32>(xsimd::neon64, uint32);

    template float32* allocateMemory<xsimd::neon64, float32>(xsimd::neon64, uint32, bool);
    template float64* allocateMemory<xsimd::neon64, float64>(xsimd::neon64, uint32, bool);
    template uint16* allocateMemory<xsimd::neon64, uint16>(xsimd::neon64, uint32, bool);
    template uint32* allocateMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32, bool);

    template float32* reallocateMemory<xsimd::neon64, float32>(xsimd::neon64, float32*, uint32, uint32);
    template float64* reallocateMemory<xsimd::neon64, float64>(xsimd::neon64, float64*, uint32, uint32);
    template uint16* reallocateMemory<xsimd::neon64, uint16>(xsimd::neon64, uint16*, uint32, uint32);
    template uint32* reallocateMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32*, uint32, uint32);

    template void freeMemory<xsimd::neon64, float32>(xsimd::neon64, float32*);
    template void freeMemory<xsimd::neon64, float64>(xsimd::neon64, float64*);
    template void freeMemory<xsimd::neon64, uint16>(xsimd::neon64, uint16*);
    template void freeMemory<xsimd::neon64, uint32>(xsimd::neon64, uint32*);
}
#endif
