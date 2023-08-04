/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #ifdef MLRLSECO_EXPORTS
        #define MLRLSECO_API __declspec(dllexport)
    #else
        #define MLRLSECO_API __declspec(dllimport)
    #endif
#else
    #define MLRLSECO_API
#endif
