/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#include "common/macros.hpp"

#include <string>

/**
 * Defines an interface for all classes that provide information about a C++ library.
 */
class MLRLCOMMON_API ILibraryInfo {
    public:

        virtual ~ILibraryInfo() {};

        /**
         * Returns the version of the C++ library.
         *
         * @return A string that specifies the library version
         */
        virtual std::string getLibraryVersion() const = 0;
};

/**
 * Returns an object of type `ILibraryVersion` that provides information about the C++ library "libmlrlcommon".
 *
 * @return A reference to an object of type `ILibraryVersion`
 */
MLRLCOMMON_API const ILibraryInfo& getCommonLibraryInfo();