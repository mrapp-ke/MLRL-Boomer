#include "common/info.hpp"

#include "config.hpp"

/**
 * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
 */
class CommonLibraryInfo final : public ILibraryInfo {
    public:

        /**
         * @see `ILibraryInfo::getLibraryName`
         */
        std::string getLibraryName() const override {
            return MLRLCOMMON_LIBRARY_NAME;
        }

        /**
         * @see `ILibraryInfo::getLibraryVersion`
         */
        std::string getLibraryVersion() const override {
            return MLRLCOMMON_LIBRARY_VERSION;
        }

        /**
         * @see `ILibraryInfo::getTargetArchitecture`
         */
        std::string getTargetArchitecture() const override {
            return MLRLCOMMON_TARGET_ARCHITECTURE;
        }

        /**
         * @see `ILibraryInfo::visitBuildOptions`
         */
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
