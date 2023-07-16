#include "seco/info.hpp"

#include "config.hpp"

namespace seco {

    /**
     * An implementation of the type `ILibraryInfo` that provides information about the C++ library "libmlrlseco".
     */
    class SeCoLibraryInfo final : public ILibraryInfo {
        public:

            std::string getLibraryName() const override {
                return MLRLSECO_LIBRARY_NAME;
            }

            std::string getLibraryVersion() const override {
                return MLRLSECO_LIBRARY_VERSION;
            }

            std::string getTargetArchitecture() const override {
                return MLRLSECO_TARGET_ARCHITECTURE;
            }
    };

    const SeCoLibraryInfo SECO_LIBRARY_INFO = SeCoLibraryInfo();

    const ILibraryInfo& getSeCoLibraryInfo() {
        return SECO_LIBRARY_INFO;
    }

}
