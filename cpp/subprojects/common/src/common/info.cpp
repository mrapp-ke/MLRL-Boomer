#include "common/info.hpp"

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
};

std::unique_ptr<ILibraryInfo> getCommonLibraryInfo() {
    return std::make_unique<CommonLibraryInfo>();
}
