#include "mlrl/common/random/rng.hpp"

const uint32 MAX_RANDOM = 0x7FFFFFFF;

RNG::RNG(uint32 randomState) : randomState_(randomState) {}

uint32 RNG::randomInt(uint32 min, uint32 max) {
    uint32* randomState = &randomState_;

    if (randomState[0] == 0) {
        randomState[0] = 1;
    }

    randomState[0] ^= static_cast<uint32>(randomState[0] << 13);
    randomState[0] ^= static_cast<uint32>(randomState[0] >> 17);
    randomState[0] ^= static_cast<uint32>(randomState[0] << 5);

    uint32 randomNumber = randomState[0] % (MAX_RANDOM + 1);
    return min + (randomNumber % (max - min));
}

bool RNG::randomBool() {
    return this->randomInt(0, 2) != 0;
}
