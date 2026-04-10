#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_qualities_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template float32 calculateOutputWiseQualities<xsimd::avx2, float32>(xsimd::avx2, const float32*, const float32*,
                                                                            const float32*, uint32, float32, float32);
        template float64 calculateOutputWiseQualities<xsimd::avx2, float64>(xsimd::avx2, const float64*, const float64*,
                                                                            const float64*, uint32, float32, float32);
    }
}
#endif
