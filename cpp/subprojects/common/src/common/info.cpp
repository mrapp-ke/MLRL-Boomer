#include "common/info.hpp"

#include "config.hpp"

/**
 * An implementation of the type `ILibraryInfo` that provides information about the C++ library "libmlrlcommon".
 */
class CommonLibraryInfo final : public ILibraryInfo {
    public:

        std::string getLibraryVersion() const override {
            return MLRLCOMMON_VERSION;
        }
};

const CommonLibraryInfo COMMON_LIBRARY_INFO = CommonLibraryInfo();

const ILibraryInfo& getCommonLibraryInfo() {
    return COMMON_LIBRARY_INFO;
}
