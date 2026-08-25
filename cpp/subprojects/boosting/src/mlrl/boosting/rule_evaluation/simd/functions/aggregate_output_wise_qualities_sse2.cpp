#include "mlrl/boosting/rule_evaluation/simd/functions/aggregate_output_wise_qualities_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template float32 aggregateOutputWiseQualities<xsimd::sse2, float32>(xsimd::sse2, const float32*, const float32*,
                                                                            const float32*, uint32, float32, float32);
        template float64 aggregateOutputWiseQualities<xsimd::sse2, float64>(xsimd::sse2, const float64*, const float64*,
                                                                            const float64*, uint32, float32, float32);

        template float32 aggregateOutputWiseQualitiesWeighted<xsimd::sse2, float32>(xsimd::sse2, const float32*,
                                                                                    const float32*, const float32*,
                                                                                    const uint32*, uint32, float32,
                                                                                    float32);
        template float64 aggregateOutputWiseQualitiesWeighted<xsimd::sse2, float64>(xsimd::sse2, const float64*,
                                                                                    const float64*, const float64*,
                                                                                    const uint32*, uint32, float32,
                                                                                    float32);
    }
}
#endif
