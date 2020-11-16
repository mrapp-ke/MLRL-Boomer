/**
 * Provides commonly used classes for handling the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "types.h"
#include <memory>


/**
 * An entry that is stored in a cache and contains an unique pointer to a vector of arbitrary type. The field
 * `numConditions` specifies how many conditions the rule contained when the vector was updated for the last time. It
 * may be used to check if the vector is still valid or must be updated.
 *
 * @tparam T The type of the vector that is stored by the entry
 */
template<class T>
struct FilteredCacheEntry {
    FilteredCacheEntry<T>() : numConditions(0) { };
    std::unique_ptr<T> vectorPtr;
    uint32 numConditions;
};
