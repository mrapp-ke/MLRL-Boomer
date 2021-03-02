/**
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */

#include "common/debugging/debug.hpp"

#pragma once

extern Debugger debugger;

// global variables are initialized to 0
extern int debugging_;
// if all debugging prints should be enabled
extern int dFull;
// print the coverage mask
extern int dCM;
// print the example weights
extern int dWeights;
// print the head score
extern int dHS;
// print which labels are covered
extern int dLC;
// print rule induction
extern int dRI;