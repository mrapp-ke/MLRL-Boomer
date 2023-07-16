#include "boosting/info.hpp"

#include "config.hpp"

namespace boosting {

    /**
     * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
     */
    class BoostingLibraryInfo final : public ILibraryInfo {
        public:

            std::string getLibraryName() const override {
                return MLRLBOOSTING_LIBRARY_NAME;
            }

            std::string getLibraryVersion() const override {
                return MLRLBOOSTING_LIBRARY_VERSION;
            }

            std::string getTargetArchitecture() const override {
                return MLRLBOOSTING_TARGET_ARCHITECTURE;
            }
    };

    const BoostingLibraryInfo BOOSTING_LIBRARY_INFO = BoostingLibraryInfo();

    const ILibraryInfo& getBoostingLibraryInfo() {
        return BOOSTING_LIBRARY_INFO;
    }

}
