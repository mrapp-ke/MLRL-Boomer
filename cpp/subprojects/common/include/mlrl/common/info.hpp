/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * Defines an interface for all classes that provide information about a C++ library.
 */
class MLRLCOMMON_API ILibraryInfo {
    public:

        /**
         * Represents a build option for configuring a library at compile-time.
         */
        struct BuildOption {
            public:

                /**
                 * @param option        The name of the build option
                 * @param description   A human-legible description of the build option
                 * @param value         The value that has been set for the build option at compile-time
                 */
                BuildOption(std::string option, std::string description, std::string value)
                    : option(option), description(description), value(value) {}

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
         * Provides information about a certain hardware resource.
         */
        struct HardwareResource {
            public:

                /**
                 * @param resource  A human-legible name of the hardware resource
                 * @param info      The information associated with the hardware resource
                 */
                HardwareResource(std::string resource, std::string info) : resource(resource), info(info) {}

                /**
                 * A human-legible name of the hardware resource.
                 */
                const std::string resource;

                /**
                 * The information associated with the hardware resource.
                 */
                const std::string info;
        };

        /**
         * A visitor function for handling objects of the type `BuildOption`.
         */
        typedef std::function<void(const BuildOption&)> BuildOptionVisitor;

        /**
         * A visitor function for handling objects of the type `HardwareResource`.
         */
        typedef std::function<void(const HardwareResource&)> HardwareResourceVisitor;

        virtual ~ILibraryInfo() {}

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
        virtual void visitBuildOptions(BuildOptionVisitor visitor) const {}

        /**
         * May be overridden by subclasses in order to invoke a given visitor function for each available hardware
         * resource.
         *
         * @param visitor A visitor function for handling objects of the type `HardwareResource`
         */
        virtual void visitHardwareResources(HardwareResourceVisitor visitor) const {}
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
 * @return True, if multi-threading support is enabled, false otherwise
 */
MLRLCOMMON_API bool isMultiThreadingSupportEnabled();

/**
 * Returns the number of CPU cores available on the machine, regardless of whether multi-threading support is enabled or
 * not.
 *
 * @return The number of CPU cores available on the machine
 */
MLRLCOMMON_API uint32 getNumCpuCores();

/**
 * Returns whether GPU support was enabled at compile-time or not.
 *
 * @return True, if GPU support is enabled, false otherwise
 */
MLRLCOMMON_API bool isGpuSupportEnabled();

/**
 * Returns whether any supported GPUs are available on the machine or not.
 *
 * @return True, if at least one supported GPU is available, false otherwise
 */
MLRLCOMMON_API bool isGpuAvailable();

/**
 * Returns the names of all supported GPUs available on the machine.
 *
 * @return An `std::vector` that contains the names of all supported GPUs
 */
MLRLCOMMON_API std::vector<std::string> getGpuDevices();
