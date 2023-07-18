/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"

#include <memory>
#include <string>

/**
 * Defines an interface for all classes that provide information about a C++ library.
 */
class MLRLCOMMON_API ILibraryInfo {
    public:

        virtual ~ILibraryInfo() {};

        /**
         * Returns the name of the C++ library.
         *
         * @return A string that specifies the library name
         */
        virtual std::string getLibraryName() const = 0;

        /**
         * Returns the version of the C++ library.
         *
         * @return A string that specifies the library version
         */
        virtual std::string getLibraryVersion() const = 0;

        /**
         * Returns the architecture that is targeted by the C++ library.
         *
         * @return A string that specifies the target architecture
         */
        virtual std::string getTargetArchitecture() const = 0;
};

/**
 * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
 *
 * @return An unique pointer to an object of type `ILibraryVersion`
 */
MLRLCOMMON_API std::unique_ptr<ILibraryInfo> getLibraryInfo();
