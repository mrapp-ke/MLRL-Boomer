#include "common/debugging/debug.hpp"
#include "common/debugging/global.hpp"
#include <iostream>

int debugging_;

void setDebugFlag() {
    debugging_ = 1;
    std::cout << "the debug flag is set\n";
}
