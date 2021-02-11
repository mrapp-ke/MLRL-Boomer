#include "debug.h"
#include "global.h"
#include <iostream>

void setDebugFlag() {
    debug_flag = 1;
    std::cout << "the debug flag is set\n";
}
