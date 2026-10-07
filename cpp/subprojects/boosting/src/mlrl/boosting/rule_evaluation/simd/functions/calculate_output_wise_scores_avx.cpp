#include "mlrl/boosting/rule_evaluation/simd/functions/calculate_output_wise_scores_generic.hpp"

#if SIMD_SUPPORT_ENABLED
namespace boosting {
    namespace simd {

        template void calculateOutputWiseScores<xsimd::avx, float32>(xsimd::avx, const float32*, const float32*,
                                                                     float32*, uint32, float32, float32);
        template void calculateOutputWiseScores<xsimd::avx, float64>(xsimd::avx, const float64*, const float64*,
                                                                     float64*, uint32, float32, float32);

        template void calculateOutputWiseScoresWeighted<xsimd::avx, float32>(xsimd::avx, const float32*, const float32*,
                                                                             const uint32*, float32*, uint32, float32,
                                                                             float32);
        template void calculateOutputWiseScoresWeighted<xsimd::avx, float64>(xsimd::avx, const float64*, const float64*,
                                                                             const uint32*, float64*, uint32, float32,
                                                                             float32);
    }
}
#endif
