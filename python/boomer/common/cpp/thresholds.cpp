#include "thresholds.h"
#include <cstddef>
#include <stdlib.h>


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
 * Adjusts the position that separates the examples that are covered by a condition from the ones that are not covered,
 * with respect to those examples that are not contained in the current sub-sample. This requires to look back a certain
 * number of examples to see if they satisfy the new condition or not. I.e., to traverse the examples in ascending or
 * descending order, depending on whether `conditionEnd` is smaller than `conditionPrevious` or vice versa, until the
 * next example that is contained in the current sub-sampling is encountered.
 *
 * @param indexedArray      A pointer to a struct of type `IndexedFloat32Array` that stores a pointer to an array which
 *                          contains the indices of the training examples and the corresponding feature values, as well
 *                          as the number of elements in the array
 * @param conditionEnd      The position that separates the covered from the uncovered examples when only taking into
 *                          account the examples that are contained in the current sub-sample
 * @param conditionPrevious The position to stop at (exclusive)
 * @param threshold         The threshold of the condition
 * @return                  The adjusted position that separates the covered from the uncovered examples with respect to
 *                          the examples that are not contained in the current sub-sample
 */
static inline intp adjustSplitOld(const IndexedFloat32Array* indexedArray, intp conditionEnd, intp conditionPrevious,
                                  float32 threshold) {
    const IndexedFloat32* indexedValues = indexedArray->data;
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
        float32 featureValue = indexedValues[r].value;
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
 * Filters a feature vector that contains the indices of the examples that are covered by the previous rule, as well as
 * their values for a certain feature, after a new condition that corresponds to said feature has been added, such that
 * the filtered vector does only contain the indices and feature values of the examples that are covered by the new
 * rule. The filtered vector is stored in a given struct of type `CacheEntry` and the given statistics are updated
 * accordingly.
 *
 * @param cacheEntry            A reference to a struct of type `CacheEntry` that should be used to store the filtered
 *                              feature vector
 * @param featureVector         A reference to an object of type `FeatureVector` that should be filtered
 * @param conditionStart        The element in `featureVector` that corresponds to the first example (inclusive)
 *                              included in the `IStatisticsSubset` that is covered by the new condition
 * @param conditionEnd          The element in `featureVector` that corresponds to the last example (exclusive)
 * @param conditionComparator   The type of the operator that is used by the new condition
 * @param covered               True, if the examples in range [conditionStart, conditionEnd) are covered by the new
 *                              condition and the remaining ones are not, false, if the examples in said range are not
 *                              covered and the remaining ones are
 * @param numConditions         The total number of conditions in the rule's body (including the new one)
 * @param coveredExamplesMask   An array of type `uint32`, shape `(num_examples)` that is used to keep track of the
 *                              indices of the examples that are covered by the previous rule. It will be updated by
 *                              this function
 * @param coveredExamplesTarget The value that is used to mark those elements in `coveredExamplesMask` that are covered
 *                              by the previous rule
 * @param statistics            A reference to an object of type `AbstractStatistics` to be notified about the examples
 *                              that must be considered when searching for the next refinement, i.e., the examples that
 *                              are covered by the new rule
 * @param weights               A reference to an an object of type `IWeightVector` that provides access to the weights
 *                              of the training examples
 * @return                      The value that is used to mark those elements in the updated `coveredExamplesMask` that
 *                              are covered by the new rule
 */
static inline uint32 filterCurrentFeatureVector(IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                                                FeatureVector& featureVector, intp conditionStart, intp conditionEnd,
                                                Comparator conditionComparator, bool covered, uint32 numConditions,
                                                uint32* coveredExamplesMask, uint32 coveredExamplesTarget,
                                                AbstractStatistics& statistics, IWeightVector& weights) {
    uint32 numTotalElements = featureVector.getNumElements();
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    bool descending = conditionEnd < conditionStart;
    uint32 updatedTarget;

    // Determine the number of elements in the filtered vector...
    uint32 numConditionSteps = abs(conditionStart - conditionEnd);
    uint32 numElements = covered ? numConditionSteps :
        (numTotalElements > numConditionSteps ? numTotalElements - numConditionSteps : 0);

    // Allocate filtered array...
    IndexedFloat32* filteredArray = numElements > 0 ?
        (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32)) : NULL;

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
        updatedTarget = numConditions;
        statistics.resetCoveredStatistics();

        // Retain the indices at positions [conditionStart, conditionEnd) and set the corresponding values in
        // `coveredExamplesMasK` to `numConditions`, which marks them as covered (because
        // `updatedTarget == numConditions`)...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coveredExamplesMask[index] = numConditions;
            filteredArray[i].index = index;
            filteredArray[i].value = iterator[r].value;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, false);
            i += direction;
        }
    } else {
        updatedTarget = coveredExamplesTarget;
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
            // `coveredExamplesMask` untouched, such that all previously covered examples in said range are still marked
            // as covered, while previously uncovered examples are still marked as uncovered...
            uint32 numSteps = abs(start - conditionStart);

            for (uint32 j = 0; j < numSteps; j++) {
                uint32 r = start + (j * direction);
                filteredArray[i].index = iterator[r].index;
                filteredArray[i].value = iterator[r].value;
                i += direction;
            }
        }

        // Discard the indices at positions [conditionStart, conditionEnd) and set the corresponding values in
        // `coveredExamplesMask` to `numConditions`, which marks them as uncovered (because
        // `updatedTarget != numConditions`)...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coveredExamplesMask[index] = numConditions;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, true);
        }

        // Retain the indices at positions [conditionEnd, end), while leaving the corresponding values in
        // `coveredExamplesMask` untouched, such that all previously covered examples in said range are still marked as
        // covered, while previously uncovered examples are still marked as uncovered...
        uint32 numSteps = abs(conditionEnd - end);

        for (uint32 j = 0; j < numSteps; j++) {
            uint32 r = conditionEnd + (j * direction);
            filteredArray[i].index = iterator[r].index;
            filteredArray[i].value = iterator[r].value;
            i += direction;
        }
    }

    IndexedFloat32Array* filteredIndexedArray = indexedArrayWrapper->array;

    if (filteredIndexedArray == NULL) {
        filteredIndexedArray = (IndexedFloat32Array*) malloc(sizeof(IndexedFloat32Array));
        indexedArrayWrapper->array = filteredIndexedArray;
    } else {
        free(filteredIndexedArray->data);
    }

    filteredIndexedArray->data = filteredArray;
    filteredIndexedArray->numElements = numElements;
    indexedArrayWrapper->numConditions = numConditions;
    return updatedTarget;
}

/**
 * Filters a feature vector that contains the indices of training examples, as well as their values for a certain
 * feature, such that the filtered vector does only contain the indices and feature values of those examples that are
 * covered by the current rule. The filtered vector is stored in a given struct of type `CacheEntry`.
 *
 * @param indexedArray          A reference to an object of type `FeatureVector` that should be filtered
 * @param cacheEntry            A reference to a struct of type `CacheEntry` that should be used to store the filtered
 *                              vector
 * @param numConditions         The total number of conditions in the current rule's body
 * @param coveredExamplesMask   An array of type `uint32`, shape `(num_examples)`, that is used to keep track of the
 *                              indices of the examples that are covered by the current rule
 * @param coveredExamplesTarget The value that is used to mark those elements in `coveredExamplesMask` that are covered
 *                              by the current rule
 */
static inline void filterAnyFeatureVector(FeatureVector& featureVector, IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                                          uint32 numConditions, const uint32* coveredExamplesMask,
                                          uint32 coveredExamplesTarget) {
    uint32 maxElements = featureVector.getNumElements();
    IndexedFloat32Array* filteredIndexedArray = indexedArrayWrapper->array;
    IndexedFloat32* filteredArray = filteredIndexedArray == NULL ? NULL : filteredIndexedArray->data;

    FeatureVector::const_iterator iterator = featureVector.cbegin();
    uint32 i = 0;

    if (maxElements > 0) {
        if (filteredArray == NULL) {
            filteredArray = (IndexedFloat32*) malloc(maxElements * sizeof(IndexedFloat32));
        }

        for (uint32 r = 0; r < maxElements; r++) {
            uint32 index = iterator[r].index;

            if (coveredExamplesMask[index] == coveredExamplesTarget) {
                filteredArray[i].index = index;
                filteredArray[i].value = iterator[r].value;
                i++;
            }
        }
    }

    if (i == 0) {
        free(filteredArray);
        filteredArray = NULL;
    } else if (i < maxElements) {
        filteredArray = (IndexedFloat32*) realloc(filteredArray, i * sizeof(IndexedFloat32));
    }

    if (filteredIndexedArray == NULL) {
        filteredIndexedArray = (IndexedFloat32Array*) malloc(sizeof(IndexedFloat32Array));
    }

    filteredIndexedArray->data = filteredArray;
    filteredIndexedArray->numElements = i;
    indexedArrayWrapper->array = filteredIndexedArray;
    indexedArrayWrapper->numConditions = numConditions;
}

AbstractThresholds::AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                       std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                       std::shared_ptr<AbstractStatistics> statisticsPtr)
    : featureMatrixPtr_(featureMatrixPtr), nominalFeatureVectorPtr_(nominalFeatureVectorPtr),
      statisticsPtr_(statisticsPtr) {

}

uint32 AbstractThresholds::getNumRows() const {
    return featureMatrixPtr_->getNumRows();
}

uint32 AbstractThresholds::getNumCols() const {
    return featureMatrixPtr_->getNumCols();
}

uint32 AbstractThresholds::getNumLabels() const {
    return statisticsPtr_->getNumCols();
}

ExactThresholdsImpl::ThresholdsSubsetImpl::ThresholdsSubsetImpl(ExactThresholdsImpl& thresholds,
                                                                std::shared_ptr<IWeightVector> weightsPtr)
    : thresholds_(thresholds), weightsPtr_(weightsPtr) {
    sumOfWeights_ = weightsPtr->getSumOfWeights();
    uint32 numExamples = thresholds.getNumRows();
    coveredExamplesMask_ = new uint32[numExamples]{0};
    coveredExamplesTarget_ = 0;
    numRefinements_ = 0;
}

ExactThresholdsImpl::ThresholdsSubsetImpl::~ThresholdsSubsetImpl() {
    std::unordered_map<uint32, IndexedFloat32ArrayWrapper*>::iterator iterator;

    for (iterator = cacheFiltered_.begin(); iterator != cacheFiltered_.end(); iterator++) {
        IndexedFloat32ArrayWrapper* indexedArrayWrapper = iterator->second;
        IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;

        if (indexedArray != NULL) {
            free(indexedArray->data);
            free(indexedArray);
        }

        free(indexedArrayWrapper);
    }

    delete[] coveredExamplesMask_;
}

std::unique_ptr<AbstractRuleRefinement> ExactThresholdsImpl::ThresholdsSubsetImpl::createRuleRefinement(
        uint32 featureIndex) {
    IndexedFloat32ArrayWrapper* indexedArrayWrapper = cacheFiltered_[featureIndex];

    if (indexedArrayWrapper == NULL) {
        indexedArrayWrapper = (IndexedFloat32ArrayWrapper*) malloc(sizeof(IndexedFloat32ArrayWrapper));
        indexedArrayWrapper->array = NULL;
        indexedArrayWrapper->numConditions = 0;
        cacheFiltered_[featureIndex] = indexedArrayWrapper;
    }

    IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;

    if (indexedArray == NULL) {
        thresholds_.cacheNew_.emplace(featureIndex, std::unique_ptr<FeatureVector>());
    }

    bool nominal = thresholds_.nominalFeatureVectorPtr_->getValue(featureIndex);
    std::unique_ptr<IRuleRefinementCallback<FeatureVector>> callbackPtr = std::make_unique<Callback>(*this);
    return std::make_unique<ExactRuleRefinementImpl>(thresholds_.statisticsPtr_, weightsPtr_, sumOfWeights_,
                                                     featureIndex, nominal, std::move(callbackPtr));
}

void ExactThresholdsImpl::ThresholdsSubsetImpl::applyRefinement(Refinement& refinement) {
    numRefinements_++;
    sumOfWeights_ = refinement.coveredWeights;

    uint32 featureIndex = refinement.featureIndex;
    IndexedFloat32ArrayWrapper* indexedArrayWrapper = cacheFiltered_[featureIndex];
    IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;

    if (indexedArray == NULL) {
        auto it = thresholds_.cacheNew_.find(featureIndex);
        FeatureVector* featureVector = it->second.get();

        if (weightsPtr_->hasZeroWeights() && abs(refinement.previous - refinement.end) > 1) {
            refinement.end = adjustSplit(*featureVector, refinement.end, refinement.previous, refinement.threshold);
        }

        coveredExamplesTarget_ = filterCurrentFeatureVector(indexedArrayWrapper, *featureVector, refinement.start,
                                                            refinement.end, refinement.comparator, refinement.covered,
                                                            numRefinements_, coveredExamplesMask_,
                                                            coveredExamplesTarget_, *thresholds_.statisticsPtr_,
                                                            *weightsPtr_);
    } else {
        // TODO Remove
        FeatureVector* featureVector = new FeatureVector(indexedArray->numElements);
        FeatureVector::iterator iterator = featureVector->begin();

        for (uint32 i = 0; i < indexedArray->numElements; i++) {
            iterator[i].index = indexedArray->data[i].index;
            iterator[i].value = indexedArray->data[i].value;
        }

        // If there are examples with zero weights, those examples have not been considered considered when searching for
        // the refinement. In the next step, we need to identify the examples that are covered by the refined rule,
        // including those that have previously been ignored, via the function `filterCurrentFeatureValues`. Said function
        // calculates the number of covered examples based on the variable `refinement.end`, which represents the position
        // that separates the covered from the uncovered examples. However, when taking into account the examples with zero
        // weights, this position may differ from the current value of `refinement.end` and therefore must be adjusted...
        // TODO Replace with adjustSplit function
        if (weightsPtr_->hasZeroWeights() && abs(refinement.previous - refinement.end) > 1) {
            refinement.end = adjustSplitOld(indexedArray, refinement.end, refinement.previous, refinement.threshold);
        }

        // Identify the examples that are covered by the refined rule...
        coveredExamplesTarget_ = filterCurrentFeatureVector(indexedArrayWrapper, *featureVector, refinement.start,
                                                      refinement.end, refinement.comparator, refinement.covered,
                                                      numRefinements_, coveredExamplesMask_, coveredExamplesTarget_,
                                                      *thresholds_.statisticsPtr_, *weightsPtr_);

        delete featureVector;
    }
}

void ExactThresholdsImpl::ThresholdsSubsetImpl::recalculatePrediction(IHeadRefinement& headRefinement,
                                                                      Refinement& refinement) const {
    PredictionCandidate& head = *refinement.headPtr;
    uint32 numLabelIndices = head.numPredictions_;
    const uint32* labelIndices = head.labelIndices_;
    float64* predictedScores = head.predictedScores_;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = thresholds_.statisticsPtr_->createSubset(numLabelIndices,
                                                                                                      labelIndices);
    uint32 numExamples = thresholds_.getNumRows();

    for (uint32 r = 0; r < numExamples; r++) {
        if (coveredExamplesMask_[r] == coveredExamplesTarget_) {
            statisticsSubsetPtr->addToSubset(r, 1);
        }
    }

    Prediction& prediction = headRefinement.calculatePrediction(*statisticsSubsetPtr, false, false);
    const float64* updatedScores = prediction.predictedScores_;

    for (uint32 c = 0; c < numLabelIndices; c++) {
        predictedScores[c] = updatedScores[c];
    }
}

void ExactThresholdsImpl::ThresholdsSubsetImpl::applyPrediction(Prediction& prediction) {
    uint32 numExamples = thresholds_.getNumRows();

    for (uint32 r = 0; r < numExamples; r++) {
        if (coveredExamplesMask_[r] == coveredExamplesTarget_) {
            thresholds_.statisticsPtr_->applyPrediction(r, prediction);
        }
    }
}

ExactThresholdsImpl::ThresholdsSubsetImpl::Callback::Callback(ThresholdsSubsetImpl& thresholdsSubset)
    : thresholdsSubset_(thresholdsSubset) {

}

ExactThresholdsImpl::ThresholdsSubsetImpl::Callback::~Callback() {
    delete featureVector_;
}

FeatureVector& ExactThresholdsImpl::ThresholdsSubsetImpl::Callback::get(uint32 featureIndex) {
    // Obtain array that contains the indices of the training examples sorted according to the current feature...
    IndexedFloat32ArrayWrapper* indexedArrayWrapper = thresholdsSubset_.cacheFiltered_[featureIndex];
    IndexedFloat32Array* indexedArray = indexedArrayWrapper->array;
    FeatureVector* featureVector;

    // TODO Remove
    IndexedFloat32Array* tmpIndexedArray = NULL;
    bool dealloc = false;

    if (indexedArray == NULL) {
        auto itFiltered = thresholdsSubset_.thresholds_.cacheNew_.find(featureIndex);
        featureVector = itFiltered->second.get();

        if (featureVector == NULL) {
            thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex, itFiltered->second);
            itFiltered->second->sortByValues();
            featureVector = itFiltered->second.get();
        }

        // TODO Remove
        tmpIndexedArray = (IndexedFloat32Array*) malloc(sizeof(IndexedFloat32Array));
        IndexedFloat32* indexedValues = NULL;

        if (featureVector->getNumElements() > 0) {
            indexedValues = (IndexedFloat32*) malloc(featureVector->getNumElements() * sizeof(IndexedFloat32));
            FeatureVector::const_iterator iterator = featureVector->cbegin();

            for (uint32 i = 0; i < featureVector->getNumElements(); i++) {
                indexedValues[i].index = iterator[i].index;
                indexedValues[i].value = iterator[i].value;
            }
        }

        tmpIndexedArray->data = indexedValues;
        tmpIndexedArray->numElements = featureVector->getNumElements();
        indexedArray = tmpIndexedArray;
    } else {
        // TODO Remove
        dealloc = true;
        featureVector = new FeatureVector(indexedArray->numElements);
        FeatureVector::iterator iterator = featureVector->begin();

        for (uint32 i = 0; i < indexedArray->numElements; i++) {
            iterator[i].index = indexedArray->data[i].index;
            iterator[i].value = indexedArray->data[i].value;
        }
    }

    // Filter feature vector, if only a subset of its elements are covered by the current rule...
    uint32 numConditions = thresholdsSubset_.numRefinements_;

    if (numConditions > indexedArrayWrapper->numConditions) {
        filterAnyFeatureVector(*featureVector, indexedArrayWrapper, numConditions,
                               thresholdsSubset_.coveredExamplesMask_, thresholdsSubset_.coveredExamplesTarget_);
        indexedArray = indexedArrayWrapper->array;
    }

    // TODO Remove
    featureVector_ = new FeatureVector(indexedArray->numElements);
    FeatureVector::iterator iterator = featureVector_->begin();

    for (uint32 i = 0; i < indexedArray->numElements; i++) {
        iterator[i].index = indexedArray->data[i].index;
        iterator[i].value = indexedArray->data[i].value;
    }

    // TODO Remove
    if (tmpIndexedArray != NULL) {
        free(tmpIndexedArray->data);
    }
    free(tmpIndexedArray);
    if (dealloc) {
        delete featureVector;
    }

    return *featureVector_;
}

ExactThresholdsImpl::ExactThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                         std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                         std::shared_ptr<AbstractStatistics> statisticsPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureVectorPtr, statisticsPtr) {

}

std::unique_ptr<IThresholdsSubset> ExactThresholdsImpl::createSubset(std::shared_ptr<IWeightVector> weightsPtr) {
    // Notify the statistics about the examples that are included in the sub-sample...
    uint32 numExamples = statisticsPtr_->getNumRows();
    statisticsPtr_->resetSampledStatistics();

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 weight = weightsPtr->getValue(r);
        statisticsPtr_->addSampledStatistic(r, weight);
    }

    return std::make_unique<ExactThresholdsImpl::ThresholdsSubsetImpl>(*this, weightsPtr);
}
