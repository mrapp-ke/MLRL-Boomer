/**
 * Provides commonly used functions for handling the thresholds that may be used by the conditions of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include <cstdlib>


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

/**
 * Updates the given statistics by applying the weights of the individual training examples.
 *
 * @param statistics    A reference to an object of type `AbstractStatistics` to be updated
 * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of the
 *                      individual training examples
 */
static inline void updateSampledStatistics(AbstractStatistics& statistics, const IWeightVector& weights) {
    uint32 numExamples = statistics.getNumRows();
    statistics.resetSampledStatistics();

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 weight = weights.getValue(r);
        statistics.addSampledStatistic(r, weight);
    }
}

/**
 * Adjusts the position that separates the examples that are covered by a condition from the ones that are not covered,
 * with respect to those examples that are not contained in the current sub-sample. This requires to look back a certain
 * number of examples to see if they satisfy the new condition or not. I.e., to traverse the examples in ascending or
 * descending order, depending on whether `conditionEnd` is smaller than `conditionPrevious` or vice versa, until the
 * next example that is contained in the current sub-sampling is encountered.
 *
 * @param featureVector     A reference to an object of type `FeatureVector` that stores the indices and feature values
 *                          of the training examples
 * @param conditionEnd      The position that separates the covered from the uncovered examples when only taking into
 *                          account the examples that are contained in the current sub-sample
 * @param conditionPrevious The position to stop at (exclusive)
 * @param threshold         The threshold of the condition
 * @return                  The adjusted position that separates the covered from the uncovered examples with respect to
 *                          the examples that are not contained in the current sub-sample
 */
static inline intp adjustSplit(FeatureVector& featureVector, intp conditionEnd, intp conditionPrevious,
                               float32 threshold) {
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    intp adjustedPosition = conditionEnd;
    bool ascending = conditionEnd < conditionPrevious;
    intp direction = ascending ? 1 : -1;
    intp start = conditionEnd + direction;
    uint32 numSteps = abs(start - conditionPrevious);

    // Traverse the examples in ascending (or descending) order until we encounter an example that is contained in the
    // current sub-sample...
    for (uint32 i = 0; i < numSteps; i++) {
        // Check if the current position should be adjusted, or not. This is the case, if the feature value of the
        // current example is smaller than or equal to the given `threshold` (or greater than the `threshold`, if we
        // traverse in descending direction)
        uint32 r = start + (i * direction);
        float32 featureValue = iterator[r].value;
        bool adjust = ascending ? featureValue <= threshold : featureValue > threshold;

        if (adjust) {
            // Update the adjusted position and continue...
            adjustedPosition = r;
        } else {
            // If we have found the first example that is separated from the example at the position we started at, we
            // are done...
            break;
        }
    }

    return adjustedPosition;
}

/**
 * Filters a given vector, containing e.g. feature values or bins, which contains the elements for a certain feature
 * that are covered by the previous rule, after a new condition that corresponds to said feature has been added, such
 * that the filtered vector does only contain the elements that are covered by the new rule. The filtered vector is
 * stored in a given struct of type `FilteredCacheEntry` and the given statistics are updated accordingly.
 *
 * @tparam T                    The type of the vector to be filtered
 * @param cacheEntry            A reference to a struct of type `FilteredCacheEntry` that should be used to store the
 *                              filtered feature vector
 * @param vector                A reference to an object of template type `T` that should be filtered
 * @param conditionStart        The element in `vector` that corresponds to the first statistic (inclusive) included in
 *                              the `IStatisticsSubset` that is covered by the new condition
 * @param conditionEnd          The element in `vector` that corresponds to the last statistic (exclusive)
 * @param conditionComparator   The type of the operator that is used by the new condition
 * @param covered               True, if the elements in range [conditionStart, conditionEnd) are covered by the new
 *                              condition and the remaining ones are not, false, if the elements in said range are not
 *                              covered, but the remaining ones are
 * @param numConditions         The total number of conditions in the rule's body (including the new one)
 * @param coverageMask          A reference to an object of type `CoverageMask` that is used to keep track of the
 *                              elements that are covered by the previous rule. It will be updated by this function
 * @param statistics            A reference to an object of type `AbstractStatistics` to be notified about the
 *                              statistics that must be considered when searching for the next refinement, i.e., the
 *                              statistics that are covered by the new rule
 * @param weights               A reference to an an object of type `IWeightVector` that provides access to the weights
 *                              of the individual training examples
 */
template<class T>
static inline void filterCurrentVector(FilteredCacheEntry<T>& cacheEntry, const T& vector, intp conditionStart,
                                       intp conditionEnd, Comparator conditionComparator, bool covered,
                                       uint32 numConditions, CoverageMask& coverageMask, AbstractStatistics& statistics,
                                       const IWeightVector& weights) {
    uint32 numTotalElements = vector.getNumElements();
    typename T::const_iterator iterator = vector.cbegin();
    bool descending = conditionEnd < conditionStart;

    // Determine the number of elements in the filtered vector...
    uint32 numConditionSteps = abs(conditionStart - conditionEnd);
    uint32 numElements = covered ? numConditionSteps :
        (numTotalElements > numConditionSteps ? numTotalElements - numConditionSteps : 0);

    // Create a new vector that will contain the filtered elements...
    std::unique_ptr<FeatureVector> filteredVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator filteredIterator = filteredVectorPtr->begin();
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    intp direction;
    uint32 i;

    if (descending) {
        direction = -1;
        i = numElements - 1;
    } else {
        direction = 1;
        i = 0;
    }

    if (covered) {
        coverageMask.target = numConditions;
        statistics.resetCoveredStatistics();

        // Retain the indices at positions [conditionStart, conditionEnd) and set the corresponding values in the given
        // `coverageMask` to `numConditions` to mark them as covered...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coverageMaskIterator[index] = numConditions;
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, false);
            i += direction;
        }
    } else {
        intp start, end;

        if (descending) {
            start = numTotalElements - 1;
            end = -1;
        } else {
            start = 0;
            end = numTotalElements;
        }

        if (conditionComparator == NEQ) {
            // Retain the indices at positions [start, conditionStart), while leaving the corresponding values in
            // `coverageMask` untouched, such that all previously covered examples in said range are still marked
            // as covered, while previously uncovered examples are still marked as uncovered...
            uint32 numSteps = abs(start - conditionStart);

            for (uint32 j = 0; j < numSteps; j++) {
                uint32 r = start + (j * direction);
                filteredIterator[i].index = iterator[r].index;
                filteredIterator[i].value = iterator[r].value;
                i += direction;
            }
        }

        // Discard the indices at positions [conditionStart, conditionEnd) and set the corresponding values in
        // `coverageMask` to `numConditions`, which marks them as uncovered...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coverageMaskIterator[index] = numConditions;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, true);
        }

        // Retain the indices at positions [conditionEnd, end), while leaving the corresponding values in `coverageMask`
        // untouched, such that all previously covered examples in said range are still marked as covered, while
        // previously uncovered examples are still marked as uncovered...
        uint32 numSteps = abs(conditionEnd - end);

        for (uint32 j = 0; j < numSteps; j++) {
            uint32 r = conditionEnd + (j * direction);
            filteredIterator[i].index = iterator[r].index;
            filteredIterator[i].value = iterator[r].value;
            i += direction;
        }
    }

    cacheEntry.vectorPtr = std::move(filteredVectorPtr);
    cacheEntry.numConditions = numConditions;
}

/**
 * Filters a given vector, containing e.g. feature values or bins, such that the filtered vector does only contain the
 * elements that are covered by the current rule. The filtered vector is stored in a given struct of type
 * `FilteredCacheEntry`.
 *
 * @tparam T            The type of the vector to be filtered
 * @param vector        A reference to an object of template type `T` that should be filtered
 * @param cacheEntry    A reference to a struct of type `FilteredCacheEntry` that should be used to store the filtered
 *                      vector
 * @param numConditions The total number of conditions in the current rule's body
 * @param coverageMask  A reference to an object of type `CoverageMask` that is used to keep track of the elements that
 *                      are covered by the current rule
 */
template<class T>
static inline void filterAnyVector(T& vector, FilteredCacheEntry<T>& cacheEntry, uint32 numConditions,
                                   const CoverageMask& coverageMask) {
    uint32 maxElements = vector.getNumElements();
    T* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::move(std::make_unique<T>(maxElements));
        filteredVector = cacheEntry.vectorPtr.get();
    }

    typename T::const_iterator iterator = vector.cbegin();
    typename T::iterator filteredIterator = filteredVector->begin();
    uint32 i = 0;

    for (uint32 r = 0; r < maxElements; r++) {
        uint32 index = iterator[r].index;

        if (coverageMask.isCovered(index)) {
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            i++;
        }
    }

    filteredVector->setNumElements(i);
    cacheEntry.numConditions = numConditions;
}