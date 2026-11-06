#include "mlrl/common/simd/functions/memory_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template uint32 getPadding<xsimd::avx512f, float32>(xsimd::avx512f, uint32);
    template uint32 getPadding<xsimd::avx512f, float64>(xsimd::avx512f, uint32);
    template uint32 getPadding<xsimd::avx512f, uint16>(xsimd::avx512f, uint32);
    template uint32 getPadding<xsimd::avx512f, uint32>(xsimd::avx512f, uint32);

    template float32* allocateMemory<xsimd::avx512f, float32>(xsimd::avx512f, uint32, bool);
    template float64* allocateMemory<xsimd::avx512f, float64>(xsimd::avx512f, uint32, bool);
    template uint16* allocateMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint32, bool);
    template uint32* allocateMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32, bool);

    template float32* reallocateMemory<xsimd::avx512f, float32>(xsimd::avx512f, float32*, uint32, uint32);
    template float64* reallocateMemory<xsimd::avx512f, float64>(xsimd::avx512f, float64*, uint32, uint32);
    template uint16* reallocateMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint16*, uint32, uint32);
    template uint32* reallocateMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*, uint32, uint32);

    template void freeMemory<xsimd::avx512f, float32>(xsimd::avx512f, float32*);
    template void freeMemory<xsimd::avx512f, float64>(xsimd::avx512f, float64*);
    template void freeMemory<xsimd::avx512f, uint16>(xsimd::avx512f, uint16*);
    template void freeMemory<xsimd::avx512f, uint32>(xsimd::avx512f, uint32*);
}
#endif
