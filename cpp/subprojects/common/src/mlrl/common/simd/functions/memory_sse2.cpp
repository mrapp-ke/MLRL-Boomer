#include "mlrl/common/simd/functions/memory_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template uint32 getPadding<xsimd::sse2, float32>(xsimd::sse2, uint32);
    template uint32 getPadding<xsimd::sse2, float64>(xsimd::sse2, uint32);
    template uint32 getPadding<xsimd::sse2, uint16>(xsimd::sse2, uint32);
    template uint32 getPadding<xsimd::sse2, uint32>(xsimd::sse2, uint32);

    template float32* allocateMemory<xsimd::sse2, float32>(xsimd::sse2, uint32, bool);
    template float64* allocateMemory<xsimd::sse2, float64>(xsimd::sse2, uint32, bool);
    template uint16* allocateMemory<xsimd::sse2, uint16>(xsimd::sse2, uint32, bool);
    template uint32* allocateMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32, bool);

    template float32* reallocateMemory<xsimd::sse2, float32>(xsimd::sse2, float32*, uint32, uint32);
    template float64* reallocateMemory<xsimd::sse2, float64>(xsimd::sse2, float64*, uint32, uint32);
    template uint16* reallocateMemory<xsimd::sse2, uint16>(xsimd::sse2, uint16*, uint32, uint32);
    template uint32* reallocateMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32*, uint32, uint32);

    template void freeMemory<xsimd::sse2, float32>(xsimd::sse2, float32*);
    template void freeMemory<xsimd::sse2, float64>(xsimd::sse2, float64*);
    template void freeMemory<xsimd::sse2, uint16>(xsimd::sse2, uint16*);
    template void freeMemory<xsimd::sse2, uint32>(xsimd::sse2, uint32*);
}
#endif
