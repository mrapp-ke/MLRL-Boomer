/**
 * Provides type definitions of tuples, as well as corresponding utility functions.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
 * value in an array.
 */
struct IndexedFloat32 {
    uint32 index;
    float32 value;
};

/**
 * A struct that contains a pointer to an array of type `IndexedFloat32`. The attribute `numElements` specifies how many
 * elements the array contains.
 */
struct IndexedFloat32Array {
    IndexedFloat32* data;
    uint32 numElements;
};

/**
 * A struct that contains a pointer to a struct of type `IndexedFloat32Array`, representing the indices and feature
 * values of the training examples that are covered by a rule. The attribute `numConditions` specifies how many
 * conditions the rule contained when the array was updated for the last time. It may be used to check if the array is
 * still valid or must be updated.
 */
struct IndexedFloat32ArrayWrapper {
    IndexedFloat32Array* array;
    uint32 numConditions;
};

/**
 * A struct that stores a value of type `float64` and a corresponding index that refers to the (original) position of
 * the value in an array.
 */
struct IndexedFloat64 {
    uint32 index;
    float64 value;
};

/**
 *  A struct that stores all necessary information of a group of examples to calculate thresholds.
 */
struct Bin {
    uint32 numExamples;
    float32 minValue;
    float32 maxValue;
};

/**
 * A structs that contains a pointer to an array of type `Bin`. The attribute `numBins` specifies how many elements the
 * array contains.
 */
struct BinArray {
    uint32 numBins;
    Bin* bins;
};

namespace tuples {

    /**
     * Compares the values of two structs of type `IndexedFloat32`.
     *
     * @param a A pointer to the first struct
     * @param b A pointer to the second struct
     * @return  -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
     *          equal, or 1 if the value of the first struct is greater than the value of the second struct
     */
    static inline int compareIndexedFloat32(const void* a, const void* b) {
        float32 v1 = ((IndexedFloat32*) a)->value;
        float32 v2 = ((IndexedFloat32*) b)->value;
        return v1 < v2 ? -1 : (v1 == v2 ? 0 : 1);
    }

    /**
     * Compares the values of two structs of type `IndexedFloat64`.
     *
     * @param a A pointer to the first struct
     * @param b A pointer to the second struct
     * @return  -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
     *          equal, or 1 if the value of the first struct is greater than the value of the second struct
     */
    static inline int compareIndexedFloat64(const void* a, const void* b) {
        float64 v1 = ((IndexedFloat64*) a)->value;
        float64 v2 = ((IndexedFloat64*) b)->value;
        return v1 < v2 ? -1 : (v1 == v2 ? 0 : 1);
    }

}
