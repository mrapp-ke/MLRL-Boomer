#include "common/info.hpp"

#include "common/util/threads.hpp"
#include "config.hpp"

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
        }
};

std::unique_ptr<ILibraryInfo> getLibraryInfo() {
    return std::make_unique<CommonLibraryInfo>();
}

bool isMultiThreadingSupportEnabled() {
    return MULTI_THREADING_SUPPORT_ENABLED ? true : false;
}

uint32 getNumCpuCores() {
    return getNumAvailableCpuCores();
}
