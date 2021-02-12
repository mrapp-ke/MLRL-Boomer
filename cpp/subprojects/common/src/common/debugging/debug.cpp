#include "common/debugging/debug.hpp"
#include "common/debugging/global.hpp"
#include <iostream>

void setDebugFlag() {
    debug_flag = 1;
    std::cout << "the debug flag is set\n";
}
