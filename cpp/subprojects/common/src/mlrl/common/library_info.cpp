#include "mlrl/common/library_info.hpp"

#include "config.hpp"
#include "mlrl/common/util/opencl.hpp"
#include "mlrl/common/util/threads.hpp"
#include "mlrl/common/util/xsimd.hpp"

static inline std::string formatVector(const std::vector<std::string>& vector) {
    if (!vector.empty()) {
        std::string result = "";

        for (auto element : vector) {
            if (!result.empty()) {
                result += ", ";
            }

            result += element;
        }

        return result;
    }

    return "none";
}

/**
 * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
 */
class CommonLibraryInfo final : public ILibraryInfo {
    public:

        std::string getLibraryName() const override {
            return MLRLCOMMON_LIBRARY_NAME;
        }

        std::string getLibraryVersion() const override {
            return MLRLCOMMON_LIBRARY_VERSION;
        }

        std::string getTargetArchitecture() const override {
            return MLRLCOMMON_TARGET_ARCHITECTURE;
        }

        void visitBuildOptions(BuildOptionVisitor visitor) const override {
            BuildOption multiThreadingBuildOption("MULTI_THREADING_SUPPORT_ENABLED", "multi-threading support",
                                                  MULTI_THREADING_SUPPORT_ENABLED ? "enabled" : "disabled");
            visitor(multiThreadingBuildOption);

            BuildOption gpuBuildOption("GPU_SUPPORT_ENABLED", "GPU support",
                                       GPU_SUPPORT_ENABLED ? "enabled" : "disabled");
            visitor(gpuBuildOption);

            BuildOption simdBuildOption("SIMD_SUPPORT_ENABLED", "SIMD support",
                                        SIMD_SUPPORT_ENABLED ? "enabled" : "disabled");
            visitor(simdBuildOption);
        }

        void visitHardwareResources(HardwareResourceVisitor visitor) const override {
            if (MULTI_THREADING_SUPPORT_ENABLED) {
                HardwareResource cpuHardwareResource("available CPU cores", std::to_string(getNumCpuCores()));
                visitor(cpuHardwareResource);
            }

            if (GPU_SUPPORT_ENABLED) {
                HardwareResource gpuHardwareResource("supported GPU devices",
                                                     formatVector(util::getSupportedGpuDevices()));
                visitor(gpuHardwareResource);
            }

            if (SIMD_SUPPORT_ENABLED) {
                HardwareResource simdHardwareResource("supported SIMD extensions",
                                                      formatVector(util::getSupportedSimdExtensions()));
                visitor(simdHardwareResource);
            }
        }
};

std::unique_ptr<ILibraryInfo> getLibraryInfo() {
    return std::make_unique<CommonLibraryInfo>();
}

bool isMultiThreadingSupportEnabled() {
    return MULTI_THREADING_SUPPORT_ENABLED ? true : false;
}

uint32 getNumCpuCores() {
    return util::getNumAvailableCpuCores();
}

bool isGpuSupportEnabled() {
    return GPU_SUPPORT_ENABLED ? true : false;
}

bool isGpuAvailable() {
    return !util::getSupportedGpuDevices().empty();
}

std::vector<std::string> getGpuDevices() {
    return util::getSupportedGpuDevices();
}

bool isSimdSupportEnabled() {
    return SIMD_SUPPORT_ENABLED ? true : false;
}

std::vector<std::string> getSupportedSimdExtensions() {
    return util::getSupportedSimdExtensions();
}
