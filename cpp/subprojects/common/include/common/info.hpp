/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
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
                 * @param o The name of the build option
                 * @param d A human-legible description of the build option
                 * @param v The value that has been set for the build option at compile-time
                 */
                BuildOption(std::string o, std::string d, std::string v) : option(o), description(d), value(v) {}

                /**
                 * The name of the build option.
                 */
                const std::string option;

                /**
                 * A human-legible description of the build option.
                 */
                const std::string description;

                /**
                 * The value that has been set for the build option at compile-time.
                 */
                const std::string value;
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
         * May be overridden by subclasses in order to invoke a given visitor function for each available build option.
         *
         * @param visitor A visitor function for handling objects of the type `BuildOption`
         */
        virtual void visitBuildOptions(BuildOptionVisitor visitor) const {};
};

/**
 * Returns an object of type `ILibraryVersion` that provides information about this C++ library.
 *
 * @return An unique pointer to an object of type `ILibraryVersion`
 */
MLRLCOMMON_API std::unique_ptr<ILibraryInfo> getLibraryInfo();

/**
 * Returns whether multi-threading support was enabled at compile-time or not.
 *
 * @return True if multi-threading support is enabled, false otherwise
 */
MLRLCOMMON_API bool isMultiThreadingSupportEnabled();

/**
 * Returns the number of CPU cores available on the machine, regardless of whether multi-threading support is enabled or
 * not.
 *
 * @return The number of CPU cores available on the machine
 */
MLRLCOMMON_API uint32 getNumCpuCores();
