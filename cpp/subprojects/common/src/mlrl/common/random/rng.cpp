#include "mlrl/common/random/rng.hpp"

#include "mlrl/common/util/validation.hpp"

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

RNGFactory::RNGFactory(uint32 randomState) : randomState_(randomState) {}

std::unique_ptr<RNG> RNGFactory::create() const {
    return std::make_unique<RNG>(randomState_);
}

RNGConfig::RNGConfig() : randomState_(1) {}

uint32 RNGConfig::getRandomState() const {
    return randomState_;
}

RNGConfig& RNGConfig::setRandomState(uint32 randomState) {
    util::assertGreaterOrEqual<uint32>("randomState", randomState, 1);
    randomState_ = randomState;
    return *this;
}

std::unique_ptr<RNGFactory> RNGConfig::createRNGFactory() const {
    return std::make_unique<RNGFactory>(randomState_);
}
