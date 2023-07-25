/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"

#include <functional>
#include <memory>
#include <string>

/**
 * Defines an interface for all classes that provide information about a C++ library.
 */
class MLRLCOMMON_API ILibraryInfo {
    public:

        /**
         * Represents a build option for configuring a library at compile-time.
         */
        struct BuildOption {
                /**
                 * The name of the build option.
                 */
                std::string option;

                /**
                 * A human-legible description of the build option.
                 */
                std::string description;

                /**
                 * The value that has been set for the build option at compile-time.
                 */
                std::string value;
        };

        /**
         * A visitor function for handling objects of the type `BuildOption`.
         */
        typedef std::function<void(const BuildOption&)> BuildOptionVisitor;

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

        /**
         * Invokes a given visitor function for each available build option.
         *
         * @param visitor A visitor function for handling objects of the type `BuildOption`
         */
        virtual void visitBuildOptions(BuildOptionVisitor visitor) const = 0;
};

/**
 * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
 *
 * @return An unique pointer to an object of type `ILibraryVersion`
 */
MLRLCOMMON_API std::unique_ptr<ILibraryInfo> getLibraryInfo();
