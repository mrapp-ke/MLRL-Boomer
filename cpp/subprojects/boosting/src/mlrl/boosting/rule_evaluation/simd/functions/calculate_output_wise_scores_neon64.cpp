#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_scores_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template void calculateOutputWiseScores<xsimd::neon64, float32>(xsimd::neon64, const float32*, const float32*,
                                                                        float32*, uint32, float32, float32);
        template void calculateOutputWiseScores<xsimd::neon64, float64>(xsimd::neon64, const float64*, const float64*,
                                                                        float64*, uint32, float32, float32);

        template void calculateOutputWiseScoresWeighted<xsimd::neon64, float32>(xsimd::neon64, const float32*,
                                                                                const float32*, const uint32*, float32*,
                                                                                uint32, float32, float32);
        template void calculateOutputWiseScoresWeighted<xsimd::neon64, float64>(xsimd::neon64, const float64*,
                                                                                const float64*, const uint32*, float64*,
                                                                                uint32, float32, float32);
    }
}
#endif
