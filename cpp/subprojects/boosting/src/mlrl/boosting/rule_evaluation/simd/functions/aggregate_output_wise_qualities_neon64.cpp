#include "mlrl/boosting/rule_evaluation/simd/functions/aggregate_output_wise_qualities_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template float32 aggregateOutputWiseQualities<xsimd::neon64, float32>(xsimd::neon64, const float32*,
                                                                              const float32*, const float32*, uint32,
                                                                              float32, float32);
        template float64 aggregateOutputWiseQualities<xsimd::neon64, float64>(xsimd::neon64, const float64*,
                                                                              const float64*, const float64*, uint32,
                                                                              float32, float32);
    }
}
#endif
