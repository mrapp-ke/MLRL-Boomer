#include "seco/info.hpp"

#include "config.hpp"

namespace seco {

    /**
     * An implementation of the type `ILibraryInfo` that provides information about this C++ library.
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

            /**
             * @see `ILibraryInfo::visitBuildOptions`
             */
            void visitBuildOptions(BuildOptionVisitor visitor) const override {}
    };

    std::unique_ptr<ILibraryInfo> getLibraryInfo() {
        return std::make_unique<SeCoLibraryInfo>();
    }

}
