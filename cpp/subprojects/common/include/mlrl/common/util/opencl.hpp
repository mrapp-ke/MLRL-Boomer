/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "config.hpp"
#include "mlrl/common/util/strings.hpp"

#include <string>
#include <vector>

#if GPU_SUPPORT_ENABLED
    #define MINIMUM_OPENCL_VERSION 120
    #define CL_HPP_MINIMUM_OPENCL_VERSION MINIMUM_OPENCL_VERSION
    #define CL_HPP_TARGET_OPENCL_VERSION MINIMUM_OPENCL_VERSION

    #include <CL/opencl.hpp>
#endif

namespace util {

#if GPU_SUPPORT_ENABLED
    /**
     * Returns the name of a specific GPU.
     *
     * @param platform  A reference to a `cl::Platform` used to access the GPU
     * @param device    A reference to a `cl::Device` representing the GPU
     * @return          The name of the GPU
     */
    static inline std::string getGpuDeviceName(const cl::Platform& platform, const cl::Device& device) {
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
        std::string gpuDeviceName = platformName + "/" + deviceName;
        return util::replaceWhitespace(util::convertToLowerCase(gpuDeviceName), "-");
    }
#endif

#if GPU_SUPPORT_ENABLED
    /**
     * Returns whether a specific GPU supports the minimum OpenCL version.
     *
     * @param device    A reference to a `cl::Device` representing the GPU
     * @return          True, if the GPU supports the minimum OpenCL version, false otherwise
     */
    static inline bool isMinimumOpenClVersionSupported(const cl::Device& device) {
        std::string minimumVersion = std::to_string(MINIMUM_OPENCL_VERSION);
        int minimumMajorVersion = std::stoi(minimumVersion.substr(0, 1));
        int minimumMinorVersion = std::stoi(minimumVersion.substr(1, 1));
        std::string deviceVersion = device.getInfo<CL_DEVICE_VERSION>();
        std::string prefix = "OpenCL ";
        std::size_t majorVersionIndex = deviceVersion.find(prefix);

        if (majorVersionIndex != std::string::npos) {
            majorVersionIndex += prefix.size();
            std::size_t minorVersionIndex = deviceVersion.find(".", majorVersionIndex);

            if (minorVersionIndex != std::string::npos && minorVersionIndex - majorVersionIndex == 1
                && minorVersionIndex + 1 < deviceVersion.size()) {
                int majorDeviceVersion = std::stoi(deviceVersion.substr(majorVersionIndex, 1));
                int minorDeviceVersion = std::stoi(deviceVersion.substr(minorVersionIndex + 1, 1));
                return majorDeviceVersion > minimumMajorVersion
                       || (majorDeviceVersion == minimumMajorVersion && minorDeviceVersion >= minimumMinorVersion);
            }
        }

        return false;
    }
#endif

    /**
     * Returns the names of all supported GPUs available on the machine.
     *
     * @return An `std::vector` that contains the names of all supported GPUs
     */
    static inline std::vector<std::string> getSupportedGpuDevices() {
        std::vector<std::string> ids;

#if GPU_SUPPORT_ENABLED
        cl::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (auto platformIterator = platforms.cbegin(); platformIterator != platforms.cend(); platformIterator++) {
            const cl::Platform& platform = *platformIterator;
            cl::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            for (auto deviceIterator = devices.cbegin(); deviceIterator != devices.cend(); deviceIterator++) {
                const cl::Device& device = *deviceIterator;

                if (isMinimumOpenClVersionSupported(device)) {
                    ids.emplace_back(getGpuDeviceName(platform, device));
                }
            }
        }
#endif

        return ids;
    }

}
