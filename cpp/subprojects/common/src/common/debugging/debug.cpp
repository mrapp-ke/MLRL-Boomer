#include "common/debugging/debug.hpp"
#include "common/debugging/global.hpp"

int debugging_;
int dFull;
int dCM;
int dWeights;
int dHS;
int dLC;
int dRI;


void setFullFlag() {
    debugging_ = 1;
    dFull = 1;
}

void setCMFlag() {
    debugging_ = 1;
    dCM = 1;
}

void setWeightsFlag() {
    debugging_ = 1;
    dWeights = 1;
}

void setHSFlag() {
    debugging_ = 1;
    dHS = 1;
}

void setLCFlag() {
    debugging_ = 1;
    dLC = 1;
}

void setRIFlag() {
    debugging_ = 1;
    dRI = 1;
}