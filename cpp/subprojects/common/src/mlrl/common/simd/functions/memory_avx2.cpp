#include "mlrl/common/simd/functions/memory_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace simd {

    template float32* allocateMemory<xsimd::avx2, float32>(xsimd::avx2, uint32, bool);
    template float64* allocateMemory<xsimd::avx2, float64>(xsimd::avx2, uint32, bool);
    template uint16* allocateMemory<xsimd::avx2, uint16>(xsimd::avx2, uint32, bool);
    template uint32* allocateMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32, bool);

    template float32* reallocateMemory<xsimd::avx2, float32>(xsimd::avx2, float32*, uint32, uint32);
    template float64* reallocateMemory<xsimd::avx2, float64>(xsimd::avx2, float64*, uint32, uint32);
    template uint16* reallocateMemory<xsimd::avx2, uint16>(xsimd::avx2, uint16*, uint32, uint32);
    template uint32* reallocateMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32*, uint32, uint32);

    template void freeMemory<xsimd::avx2, float32>(xsimd::avx2, float32*);
    template void freeMemory<xsimd::avx2, float64>(xsimd::avx2, float64*);
    template void freeMemory<xsimd::avx2, uint16>(xsimd::avx2, uint16*);
    template void freeMemory<xsimd::avx2, uint32>(xsimd::avx2, uint32*);
}
#endif
