#include "tuples.h"
#include <limits>


namespace tuples {

    /**
     * Compares the values of two structs of type `IndexedValue`.
     *
     * @param a A pointer to the first struct
     * @param b A pointer to the second struct
     * @return  -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
     *          equal, or 1 if the value of the first struct is greater than the value of the second struct
     */
    template<class T>
    static inline int compareIndexedValue(const void* a, const void* b) {
        T v1 = ((IndexedValue<T>*) a)->value;
        T v2 = ((IndexedValue<T>*) b)->value;
        return v1 < v2 ? -1 : (v1 == v2 ? 0 : 1);
    }

}
