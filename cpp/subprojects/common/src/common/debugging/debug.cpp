#include "common/debugging/debug.hpp"
#include "common/debugging/global.hpp"
#include <iostream>

int debugging_;
int dFull;
int dCM;
int dWeights;
int dHS;
int dLC;
int dAR;


void setFullFlag() {
    debugging_ = 1;
    dFull = 1;
    std::cout << "full debug flag is set\n";
}

void setCMFlag() {
    debugging_ = 1;
    dCM = 1;
    std::cout << "coverage mask flag is set\n";
}

void setWeightsFlag() {
    debugging_ = 1;
    dWeights = 1;
    std::cout << "weights flag is set\n";
}

void setHSFlag() {
    debugging_ = 1;
    dHS = 1;
    std::cout << "head score flag is set\n";
}

void setLCFlag() {
    debugging_ = 1;
    dLC = 1;
    std::cout << "label coverage flag is set\n";
}

void setARFlag() {
    debugging_ = 1;
    dAR = 1;
    std::cout << "alternative rules flag is set\n";
}