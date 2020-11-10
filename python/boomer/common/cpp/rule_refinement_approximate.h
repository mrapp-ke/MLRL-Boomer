/**
 * Implements classes that allow to find the best refinement of rules based on approximate thresholds that result from
 * the boundaries between the bins that have been creating using a binning method.
 *
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "rule_refinement.h"
#include <forward_list>
#include <cstdlib>


/**
 * TODO
 */
class BinVectorNew {

    public:

        typedef IndexedValue<float32> Example;

    private:

        DenseVector<Bin> bins_;

        std::forward_list<Example>** examplesPerBin_;

    public:

        BinVectorNew(uint32 numElements)
            : bins_(DenseVector<Bin>(numElements, true)),
              examplesPerBin_((std::forward_list<Example>**) calloc(numElements, sizeof(std::forward_list<Example>*))) {

        }

        ~BinVectorNew() {
            uint32 numElements = this->getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                delete examplesPerBin_[i];
            }

            free(examplesPerBin_);
        }

        typedef DenseVector<Bin>::iterator iterator;

        typedef DenseVector<Bin>::const_iterator const_iterator;

        typedef std::forward_list<Example>::const_iterator example_const_iterator;

        /**
         * TODO
         *
         * @return TODO
         */
        iterator begin() {
            return bins_.begin();
        }

        /**
         * TODO
         *
         * @return TODO
         */
        iterator end() {
            return bins_.end();
        }

        /**
         * TODO
         *
         * @return TODO
         */
        const_iterator cbegin() const {
            return bins_.cbegin();
        }

        /**
         * TODO
         *
         * @return TODO
         */
        const_iterator cend() const {
            return bins_.cend();
        }

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @return          TODO
         */
        example_const_iterator examples_cbegin(uint32 binIndex) const {
            std::forward_list<Example>* examples = examplesPerBin_[binIndex];

            if (examples == nullptr) {
                examples = new std::forward_list<IndexedValue<float32>>();
            }

            return examples->cbegin();
        }

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @return          TODO
         */
        example_const_iterator examples_cend(uint32 binIndex) const {
            std::forward_list<Example>* examples = examplesPerBin_[binIndex];

            if (examples == nullptr) {
                examples = new std::forward_list<IndexedValue<float32>>();
            }

            return examples->cend();
        }

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @param example   TODO
         */
        void addExample(uint32 binIndex, Example example) {
            std::forward_list<Example>* examples = examplesPerBin_[binIndex];

            if (examples == nullptr) {
                examples = new std::forward_list<IndexedValue<float32>>();
            }

            examples->push_front(example);
        }

        /**
         * TODO
         */
        void clearAllExamples() {
             uint32 numElements = this->getNumElements();

            for (uint32 i = 0; i < numElements; i++) {
                std::forward_list<Example>* examples = examplesPerBin_[i];

                if (examples != nullptr) {
                    examples->clear();
                }
            }
        }

        /**
         * Returns the number of bins in the vector.
         *
         * @return The number of bins in the vector
         */
        uint32 getNumElements() const {
            return bins_.getNumElements();
        }

        /**
         * Sets the number of bins in the vector.
         *
         * @param numElements The number of bins to be set
         */
        void setNumElements(uint32 numElements) {
            bins_.setNumElements(numElements);

            examplesPerBin_ = (std::forward_list<Example>**) realloc(
                examplesPerBin_, numElements * sizeof(std::forward_list<Example>**));
        }

};

/**
 * TODO
 */
class BinVectorNew2 : public DenseVector<Bin> {

    public:

        typedef IndexedValue<float32> Example;

    private:

        std::unordered_map<uint32, std::forward_list<Example>> examplesPerBin_;

    public:

        /**
         * @param numElements TODO
         */
        BinVectorNew2(uint32 numElements)
            : DenseVector<Bin>(numElements, true) {

        }

        typedef std::forward_list<Example>::const_iterator example_const_iterator;

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @return          TODO
         */
        example_const_iterator examples_cbegin(uint32 binIndex) {
            std::forward_list<Example>& examples = examplesPerBin_[binIndex];
            return examples.cbegin();
        }

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @return          TODO
         */
        example_const_iterator examples_cend(uint32 binIndex) {
            std::forward_list<Example>& examples = examplesPerBin_[binIndex];
            return examples.cend();
        }

        /**
         * TODO
         *
         * @param binIndex  TODO
         * @param example   TODO
         */
        void addExample(uint32 binIndex, Example example) {
            std::forward_list<Example>& examples = examplesPerBin_[binIndex];
            examples.push_front(example);
        }

        /**
         * TODO
         */
        void clearAllExamples() {
            examplesPerBin_.clear();
        }

};

/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the boundaries between the bins
 * that have been created using a binning method.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the refined rule is
 *           allowed to predict
 */
template<class T>
class ApproximateRuleRefinement : public IRuleRefinement {

    private:

        std::unique_ptr<IHeadRefinement> headRefinementPtr_;

        const T& labelIndices_;

        uint32 featureIndex_;

        std::unique_ptr<IRuleRefinementCallback<BinVector>> callbackPtr_;

        std::unique_ptr<Refinement> refinementPtr_;

    public:

        /**
         * @param headRefinementPtr An unique pointer to an object of type `IHeadRefinement` that should be used to find
         *                          the head of refined rules
         * @param labelIndices      A reference to an object of template type `T` that provides access to the indices of
         *                          the labels for which the refined rule is allowed to predict
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param callbackPtr       An unique pointer to an object of type `IRuleRefinementCallback<BinVector>` that
         *                          allows to retrieve the bins for a certain feature
         */
        ApproximateRuleRefinement(std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices,
                                  uint32 featureIndex, std::unique_ptr<IRuleRefinementCallback<BinVector>> callbackPtr)
            : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices),
              featureIndex_(featureIndex), callbackPtr_(std::move(callbackPtr)) {

        }

        void findRefinement(const AbstractEvaluatedPrediction* currentHead) override {
            std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
            refinementPtr->featureIndex = featureIndex_;
            refinementPtr->start = 0;
            const AbstractEvaluatedPrediction* bestHead = currentHead;

            // Invoke the callback...
            std::unique_ptr<IRuleRefinementCallback<BinVector>::Result> callbackResultPtr = callbackPtr_->get();
            const AbstractStatistics& statistics = callbackResultPtr->first;
            const BinVector& binVector = callbackResultPtr->second;
            BinVector::const_iterator iterator = binVector.cbegin();
            uint32 numBins = binVector.getNumElements();

            // Create a new, empty subset of the current statistics when processing a new feature...
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

            // Search for the first non-empty bin...
            uint32 r = 0;

            while (iterator[r].numExamples == 0 && r < numBins) {
                r++;
            }

            statisticsSubsetPtr->addToSubset(r, 1);
            uint32 previousR = r;
            float32 previousValue = iterator[r].maxValue;
            uint32 numCoveredExamples = iterator[r].numExamples;

            for (r = r + 1; r < numBins; r++) {
                uint32 numExamples = iterator[r].numExamples;

                if (numExamples > 0) {
                    float32 currentValue = iterator[r].minValue;

                    const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead,
                                                                                           *statisticsSubsetPtr, false,
                                                                                           false);

                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->comparator = LEQ;
                        refinementPtr->threshold = (previousValue + currentValue) / 2.0;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = numCoveredExamples;
                        refinementPtr->covered = true;
                    }

                    head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->comparator = GR;
                        refinementPtr->threshold = (previousValue + currentValue) / 2.0;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = numCoveredExamples;
                        refinementPtr->covered = false;
                    }

                    previousValue = iterator[r].maxValue;
                    previousR = r;
                    numCoveredExamples += numExamples;
                    statisticsSubsetPtr->addToSubset(r, 1);
                }
            }

            refinementPtr->headPtr = headRefinementPtr_->pollHead();
            refinementPtr_ = std::move(refinementPtr);
        }

        std::unique_ptr<Refinement> pollRefinement() override {
            return std::move(refinementPtr_);
        }

};
