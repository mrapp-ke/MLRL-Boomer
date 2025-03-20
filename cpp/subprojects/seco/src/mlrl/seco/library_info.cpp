#include "mlrl/seco/library_info.hpp"

#include "config.hpp"

namespace seco {

    /**
     * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
     */
    class SeCoLibraryInfo final : public ILibraryInfo {
        public:

            /**
             * @see `ILibraryInfo::getLibraryName`
             */
            std::string getLibraryName() const override {
                return MLRLSECO_LIBRARY_NAME;
            }

            /**
             * @see `ILibraryInfo::getLibraryVersion`
             */
            std::string getLibraryVersion() const override {
                return MLRLSECO_LIBRARY_VERSION;
            }

            /**
             * @see `ILibraryInfo::getTargetArchitecture`
             */
            std::string getTargetArchitecture() const override {
                return MLRLSECO_TARGET_ARCHITECTURE;
            }
    };

    std::unique_ptr<ILibraryInfo> getLibraryInfo() {
        return std::make_unique<SeCoLibraryInfo>();
    }

}
