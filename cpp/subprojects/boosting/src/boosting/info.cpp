#include "boosting/info.hpp"

#include "config.hpp"

namespace boosting {

    /**
     * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
     */
    class BoostingLibraryInfo final : public ILibraryInfo {
        public:

            /**
             * @see `ILibraryInfo::getLibraryName`
             */
            std::string getLibraryName() const override {
                return MLRLBOOSTING_LIBRARY_NAME;
            }

            /**
             * @see `ILibraryInfo::getLibraryVersion`
             */
            std::string getLibraryVersion() const override {
                return MLRLBOOSTING_LIBRARY_VERSION;
            }

            /**
             * @see `ILibraryInfo::getTargetArchitecture`
             */
            std::string getTargetArchitecture() const override {
                return MLRLBOOSTING_TARGET_ARCHITECTURE;
            }

            /**
             * @see `ILibraryInfo::visitBuildOptions`
             */
            void visitBuildOptions(BuildOptionVisitor visitor) const override {}
    };

    std::unique_ptr<ILibraryInfo> getLibraryInfo() {
        return std::make_unique<BoostingLibraryInfo>();
    }

}
