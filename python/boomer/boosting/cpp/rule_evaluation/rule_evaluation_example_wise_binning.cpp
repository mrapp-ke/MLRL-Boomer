#include "rule_evaluation_example_wise_binning.h"
#include "rule_evaluation_label_wise_binning_common.h"
#include "rule_evaluation_example_wise_common.h"
#include "../../../common/cpp/rule_evaluation/score_vector_label_wise_binned_dense.h"
#include "../binning/label_binning_equal_width.h"
#include "../math/blas.h"
#include <forward_list>
#include <cstdlib>
#include <iostream>

using namespace boosting;


/**
 * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
 *
 * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the coefficients
 * @param n                         The number of rows and columns in the coefficient matrix
 * @param numElementsPerBin         A pointer to an array of type `uint32`, shape `(n)` that stores the number of
 *                                  elements per bin
 * @param l2RegularizationWeight    The L2 regularization weight to be added
 */
static inline void addRegularizationWeight(float64* output, uint32 n, const uint32* numElementsPerBin,
                                           float64 l2RegularizationWeight) {
    for (uint32 i = 0; i < n; i++) {
        float64 weight = (float64) numElementsPerBin[i];
        output[(i * n) + i] += (weight * l2RegularizationWeight);
    }
}

template<class T>
class Mapping {

    public:

        typedef std::forward_list<T> Bin;

    private:

        uint32 numBins_;

        Bin** bins_;

    public:

        Mapping(uint32 numBins)
            : numBins_(numBins), bins_((Bin**) calloc(numBins, sizeof(Bin*))) {

        }

        ~Mapping() {
            this->clear();
            free(bins_);
        }

        Bin& get(uint32 pos) {
            Bin** binPtr = &bins_[pos];

            if (*binPtr == nullptr) {
                *binPtr = new Bin();
            }

            return **binPtr;
        }

        void clear() {
            for (uint32 i = 0; i < numBins_; i++) {
                Bin** binPtr = &bins_[i];

                if (*binPtr != nullptr) {
                    delete *binPtr;
                    *binPtr = nullptr;
                }
            }
        }

};

/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 * The labels are assigned to bins based on the corresponding gradients.
 *
 * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
 */
template<class T>
class BinningExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation<T> {

    private:

        class LabelWiseBinningObserver : public IBinningObserver<float64> {

            private:

                const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation_;

            public:

                LabelWiseBinningObserver(const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation)
                    : ruleEvaluation_(ruleEvaluation) {

                }

                void onBinUpdate(uint32 binIndex, uint32 originalIndex, float64 value) override {
                    ruleEvaluation_.tmpGradients_[binIndex] += value;
                    float64 hessian =
                        ruleEvaluation_.currentStatisticVector_->hessians_diagonal_cbegin()[originalIndex];
                    ruleEvaluation_.tmpHessians_[binIndex] += hessian;
                    ruleEvaluation_.numElementsPerBin_[binIndex] += 1;
                    ruleEvaluation_.labelWiseScoreVector_->indices_binned_begin()[originalIndex] = binIndex;
                }

        };

        class ExampleWiseBinningObserver : public IBinningObserver<float64> {

            private:

                const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation_;

            public:

                ExampleWiseBinningObserver(const BinningExampleWiseRuleEvaluation<T>& ruleEvaluation)
                    : ruleEvaluation_(ruleEvaluation) {

                }

                void onBinUpdate(uint32 binIndex, uint32 originalIndex, float64 value) override {
                    ruleEvaluation_.tmpGradients_[binIndex] += value;
                    ruleEvaluation_.mapping_->get(binIndex).push_front(originalIndex);
                    ruleEvaluation_.numElementsPerBin_[binIndex] += 1;
                    ruleEvaluation_.scoreVector_->indices_binned_begin()[originalIndex] = binIndex;
                    std::cout << originalIndex << " ==> " << binIndex <<  " (numElementsPerBin = " << ruleEvaluation_.numElementsPerBin_[binIndex] << ")\n";
                }

        };

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

        std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr_;

        std::shared_ptr<Blas> blasPtr_;

        DenseBinnedScoreVector<T>* scoreVector_;

        DenseBinnedLabelWiseScoreVector<T>* labelWiseScoreVector_;

        float64* tmpGradients_;

        float64* tmpHessians_;

        uint32* numElementsPerBin_;

        Mapping<uint32>* mapping_;

        IBinningObserver<float64>* binningObserver_;

        const DenseExampleWiseStatisticVector* currentStatisticVector_;

    public:

        /**
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be used
         *                                  to assign labels to bins
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        BinningExampleWiseRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, uint32 numPositiveBins,
                                         uint32 numNegativeBins,
                                         std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr,
                                         std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation<T>(labelIndices, lapackPtr),
              l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins), binningPtr_(std::move(binningPtr)), blasPtr_(blasPtr),
              scoreVector_(nullptr), labelWiseScoreVector_(nullptr), tmpGradients_(nullptr), tmpHessians_(nullptr),
              numElementsPerBin_(nullptr), mapping_(nullptr), binningObserver_(nullptr) {

        }

        ~BinningExampleWiseRuleEvaluation() {
            delete scoreVector_;
            delete labelWiseScoreVector_;
            free(tmpGradients_);
            free(tmpHessians_);
            free(numElementsPerBin_);
            delete mapping_;
            delete binningObserver_;
        }

        const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            uint32 numBins;

            if (labelWiseScoreVector_ == nullptr) {
                numBins = numPositiveBins_ + numNegativeBins_;
                labelWiseScoreVector_ = new DenseBinnedLabelWiseScoreVector<T>(this->labelIndices_, numBins);
                tmpGradients_ = (float64*) malloc(numBins * sizeof(float64));
                tmpHessians_ = (float64*) malloc(numBins * sizeof(float64));
                numElementsPerBin_ = (uint32*) malloc(numBins * sizeof(uint32));
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::LabelWiseBinningObserver(*this);
            } else {
                numBins = labelWiseScoreVector_->getNumBins();
            }

            // Reset gradients and Hessians to zero...
            for (uint32 i = 0; i < numBins; i++) {
                tmpGradients_[i] = 0;
                tmpHessians_[i] = 0;
                numElementsPerBin_[i] = 0;
            }

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(numPositiveBins_, numNegativeBins_, statisticVector, *binningObserver_);

            // Compute predictions and quality scores...
            labelWiseScoreVector_->overallQualityScore = calculateLabelWisePredictionInternally<
                    typename DenseBinnedLabelWiseScoreVector<T>::score_binned_iterator,
                    typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator, float64*, float64*,
                    uint32*>(
                numBins, labelWiseScoreVector_->scores_binned_begin(),
                labelWiseScoreVector_->quality_scores_binned_begin(), tmpGradients_, tmpHessians_, numElementsPerBin_,
                l2RegularizationWeight_);
            return *labelWiseScoreVector_;
        }

        const IScoreVector& calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
            uint32 numBins;

            if (scoreVector_ == nullptr) {
                numBins = numPositiveBins_ + numNegativeBins_;
                scoreVector_ = new DenseBinnedScoreVector<T>(this->labelIndices_, numBins);
                this->initializeTmpArrays(numBins);
                tmpGradients_ = (float64*) malloc(numBins * sizeof(float64));
                tmpHessians_ = (float64*) malloc(triangularNumber(numBins) * sizeof(float64));
                numElementsPerBin_ = (uint32*) malloc(numBins * sizeof(uint32));
                mapping_ = new Mapping<uint32>(numBins);
                binningObserver_ = new BinningExampleWiseRuleEvaluation<T>::ExampleWiseBinningObserver(*this);
            } else {
                numBins = scoreVector_->getNumBins();
            }

            // Reset gradients and Hessians to zero...
            std::cout << "reset mapping...\n";
            mapping_->clear();
            std::cout << "reset mapping...DONE\n";

            std::cout << "reset arrays to zero...\n";
            for (uint32 i = 0; i < numBins; i++) {
                tmpGradients_[i] = 0;
                numElementsPerBin_[i] = 0;
            }
            std::cout << "reset arrays to zero...DONE\n";

            // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
            std::cout << "createBins...\n";
            currentStatisticVector_ = &statisticVector;
            binningPtr_->createBins(numPositiveBins_, numNegativeBins_, statisticVector, *binningObserver_);
            std::cout << "createBins...DONE\n";

            std::cout << "create binned Hessian matrix...\n";
            DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();

            for (uint32 i = 0; i < numBins; i++) {
                uint32 offset = triangularNumber(i);

                for (uint32 j = 0; j < i + 1; j++) {
                    float64 sumOfHessians = 0;
                    Mapping<uint32>::Bin& bin1 = mapping_->get(i);
                    Mapping<uint32>::Bin& bin2 = mapping_->get(j);

                    std::cout << "B_{" << i << "," << j << "}: ";

                    for (auto it1 = bin1.cbegin(); it1 != bin1.cend(); it1++) {
                        for (auto it2 = bin2.cbegin(); it2 != bin2.cend(); it2++) {
                            uint32 index1 = *it1;
                            uint32 index2 = *it2;
                            uint32 r, c;

                            if (index1 < index2) {
                                r = index1;
                                c = index2;
                            } else {
                                r = index2;
                                c = index1;
                            }

                            std::cout << "H_{" << r << ", " << c << "} + ";
                            sumOfHessians += hessianIterator[triangularNumber(c) + r];
                        }
                    }

                    std::cout << "\n";

                    tmpHessians_[offset + j] = sumOfHessians;
                }
            }
            std::cout << "create binned Hessian matrix...DONE\n";

            std::cout << "---------------------------------------------------------------------\n";
            std::cout << "numElementsPerBin_:\n";
            std::cout << "---------------------------------------------------------------------\n";
            for (uint32 i = 0; i < numBins; i++) {
                std::cout << i << " = " << numElementsPerBin_[i] << "\n";
            }

            std::cout << "---------------------------------------------------------------------\n";
            std::cout << "tmpGradients_:\n";
            std::cout << "---------------------------------------------------------------------\n";
            for (uint32 i = 0; i < numBins; i++) {
                std::cout << i << " = " << tmpGradients_[i] << "\n";
            }

            std::cout << "---------------------------------------------------------------------\n";
            std::cout << "indices_binned:\n";
            std::cout << "---------------------------------------------------------------------\n";
            for (uint32 i = 0; i < scoreVector_->getNumElements(); i++) {
                std::cout << i << " = " << scoreVector_->indices_binned_begin()[i] << "\n";
            }

            std::cout << "---------------------------------------------------------------------\n";
            std::cout << "tmpHessians_:\n";
            std::cout << "---------------------------------------------------------------------\n";
            for (uint32 i = 0; i < triangularNumber(numBins); i++) {
                std::cout << i << " = " << tmpHessians_[i] << "\n";
            }

            std::cout << "---------------------------------------------------------------------\n";
            std::cout << "mapping_:\n";
            std::cout << "---------------------------------------------------------------------\n";
            for (uint32 i = 0; i < numBins; i++) {
                Mapping<uint32>::Bin& bin = mapping_->get(i);
                std::cout << i << ": ";

                for (auto it = bin.cbegin(); it != bin.cend(); it++) {
                    std::cout << *it << ", ";
                }

                std::cout << "\n";
            }

            std::cout << "---------------------------------------------------------------------\n";

            typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                scoreVector_->scores_binned_begin();
            std::cout << "copyCoefficients...\n";
            copyCoefficients<float64*>(tmpHessians_, this->dsysvTmpArray1_, numBins);
            std::cout << "copyCoefficients...DONE\n";
            std::cout << "addRegularizationWeight...\n";
            addRegularizationWeight(this->dsysvTmpArray1_, numBins, numElementsPerBin_, l2RegularizationWeight_);
            std::cout << "addRegularizationWeight...DONE\n";
            std::cout << "copyOrdinates...\n";
            copyOrdinates<float64*>(tmpGradients_, scoreIterator, numBins);
            std::cout << "copyOrdinates...DONE\n";

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            std::cout << "dsysv...\n";
            this->lapackPtr_->dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                                    numBins, this->dsysvLwork_);
            std::cout << "dsysv...DONE\n";

            // Calculate the overall quality score...
            std::cout << "calculateExampleWiseQualityScore...\n";
            float64 qualityScore = calculateExampleWiseQualityScore(numBins, scoreIterator, tmpGradients_, tmpHessians_,
                                                                    *blasPtr_, this->dspmvTmpArray_);
            std::cout << "calculateExampleWiseQualityScore...DONE\n";
            std::cout << "add regularization weight...\n";
            qualityScore += 0.5 * l2RegularizationWeight_ *
                            l2NormPow<typename DenseBinnedScoreVector<T>::score_binned_iterator, uint32*>(
                                scoreIterator, numElementsPerBin_, numBins);
            std::cout << "add regularization weight...DONE\n";
            scoreVector_->overallQualityScore = qualityScore;
            return *scoreVector_;
        }

};

EqualWidthBinningExampleWiseRuleEvaluationFactory::EqualWidthBinningExampleWiseRuleEvaluationFactory(
        float64 l2RegularizationWeight, uint32 numPositiveBins, uint32 numNegativeBins, std::shared_ptr<Blas> blasPtr,
        std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const FullIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>();
    return std::make_unique<BinningExampleWiseRuleEvaluation<FullIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                               numPositiveBins_, numNegativeBins_,
                                                                               std::move(binningPtr), blasPtr_,
                                                                               lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> EqualWidthBinningExampleWiseRuleEvaluationFactory::create(
        const PartialIndexVector& indexVector) const {
    std::unique_ptr<ILabelBinning<DenseExampleWiseStatisticVector>> binningPtr =
        std::make_unique<EqualWidthLabelBinning<DenseExampleWiseStatisticVector>>();
    return std::make_unique<BinningExampleWiseRuleEvaluation<PartialIndexVector>>(indexVector, l2RegularizationWeight_,
                                                                                  numPositiveBins_, numNegativeBins_,
                                                                                  std::move(binningPtr), blasPtr_,
                                                                                  lapackPtr_);
}
