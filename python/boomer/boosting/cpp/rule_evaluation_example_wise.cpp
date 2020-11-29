#include "rule_evaluation_example_wise.h"
#include "math/math.h"
#include <cstdlib>
#include <cmath>
#include <limits>

using namespace boosting;


/**
 * Copies the Hessians that are stored by a `DenseExampleWiseStatisticVector` to a coefficient matrix that may be passed
 * to LAPACK's DSYSV routine.
 *
 * @param statisticVector   A reference to an object of type `DenseExampleWiseStatisticVector` that stores the Hessians
 * @param output            A pointer to an array of type `float64`, shape `(n, n)`, the Hessians should be copied to
 * @param n                 The number of rows and columns in the coefficient matrix
 */
static inline void copyCoefficients(const DenseExampleWiseStatisticVector& statisticVector, float64* output, uint32 n) {
    DenseExampleWiseStatisticVector::hessian_const_iterator hessianIterator = statisticVector.hessians_cbegin();

    for (uint32 c = 0; c < n; c++) {
        uint32 offset = c * n;

        for (uint32 r = 0; r < c + 1; r++) {
            float64 hessian = *hessianIterator;
            output[offset + r] = hessian;
            hessianIterator++;
        }
    }
}

/**
 * Adds a specific L2 regularization weight to the diagonal of a coefficient matrix.
 *
 * @param output                    A pointer to an array of type `float64`, shape `(n, n)` that stores the coefficients
 * @param n                         The number of rows and columns in the coefficient matrix
 * @param l2RegularizationWeight    The L2 regularization weight to be added
 */
static inline void addRegularizationWeight(float64* output, uint32 n, float64 l2RegularizationWeight) {
    for (uint32 i = 0; i < n; i++) {
        output[(i * n) + i] += l2RegularizationWeight;
    }
}

/**
 * Copies the gradients that are stored by a `DenseExampleWiseStatisticVector` to a vector of ordinates that may be
 * passed to LAPACK's DSYSV routine.
 *
 * @param statisticVector   A reference to an object of type `DenseExampleWiseStatisticVector` that stores the gradients
 * @param output            A pointer to an array of type `float64`, shape `(n)`, the gradients should be copied to
 * @param n                 The number of ordinates
 */
static inline void copyOrdinates(const DenseExampleWiseStatisticVector& statisticVector, float64* output, uint32 n) {
    DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

    for (uint32 i = 0; i < n; i++) {
        float64 gradient = *gradientIterator;
        output[i] = -gradient;
        gradientIterator++;
    }
}

/**
 * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
 * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that is
 * applied example-wise.
 */
class AbstractExampleWiseRuleEvaluation : public IExampleWiseRuleEvaluation {

    private:

        uint32 numPredictions_;

        DenseScoreVector* scoreVector_;

        DenseLabelWiseScoreVector* labelWiseScoreVector_;

        int dsysvLwork_;

        float64* dsysvTmpArray1_;

        int* dsysvTmpArray2_;

        double* dsysvTmpArray3_;

        float64* dspmvTmpArray_;

    protected:

        std::shared_ptr<Lapack> lapackPtr_;

        virtual void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                                  DenseLabelWiseScoreVector& scoreVector) = 0;

        virtual void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                                    DenseScoreVector& scoreVector, int dsysvLwork,
                                                    float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                    double* dsysvTmpArray3, float64* dspmvTmpArray) = 0;

    public:

        /**
         * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
         *                          LAPACK routines
         * @param numPredictions    The number of labels for which the rules may predict
         */
        AbstractExampleWiseRuleEvaluation(std::shared_ptr<Lapack> lapackPtr, uint32 numPredictions)
            : numPredictions_(numPredictions), scoreVector_(nullptr), labelWiseScoreVector_(nullptr),
              dsysvTmpArray1_(nullptr), dsysvTmpArray2_(nullptr), dsysvTmpArray3_(nullptr), dspmvTmpArray_(nullptr),
              lapackPtr_(lapackPtr) {

        }

        ~AbstractExampleWiseRuleEvaluation() {
            delete scoreVector_;
            delete labelWiseScoreVector_;
            free(dsysvTmpArray1_);
            free(dsysvTmpArray2_);
            free(dsysvTmpArray3_);
            free(dspmvTmpArray_);
        }

        const DenseLabelWiseScoreVector& calculateLabelWisePrediction(
                const DenseExampleWiseStatisticVector& statisticVector) override {
            if (labelWiseScoreVector_ == nullptr) {
                labelWiseScoreVector_ = new DenseLabelWiseScoreVector(numPredictions_);
            }

            this->calculateLabelWisePrediction(statisticVector, *labelWiseScoreVector_);
            return *labelWiseScoreVector_;
        }

        const DenseScoreVector& calculateExampleWisePrediction(
                DenseExampleWiseStatisticVector& statisticVector) override {
            if (scoreVector_ == nullptr) {
                scoreVector_ = new DenseScoreVector(numPredictions_);
                dsysvTmpArray1_ = (float64*) malloc(numPredictions_ * numPredictions_ * sizeof(float64));
                dsysvTmpArray2_ = (int*) malloc(numPredictions_ * sizeof(int));
                dspmvTmpArray_ = (float64*) malloc(numPredictions_ * sizeof(float64));

                // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                dsysvLwork_ = lapackPtr_->queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions_);
                dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
            }

            this->calculateExampleWisePrediction(statisticVector, *scoreVector_, dsysvLwork_, dsysvTmpArray1_,
                                                 dsysvTmpArray2_, dsysvTmpArray3_, dspmvTmpArray_);
            return *scoreVector_;
        }

};

/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 */
class RegularizedExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        std::shared_ptr<Blas> blasPtr_;

    protected:

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          DenseLabelWiseScoreVector& scoreVector) override {
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                statisticVector.gradients_cbegin();
            uint32 numPredictions = scoreVector.getNumElements();
            DenseLabelWiseScoreVector::score_iterator scoreIterator = scoreVector.scores_begin();
            DenseLabelWiseScoreVector::quality_score_iterator qualityScoreIterator = scoreVector.quality_scores_begin();
            float64 overallQualityScore = 0;

            // For each label, calculate the score to be predicted, as well as a quality score...
            for (uint32 c = 0; c < numPredictions; c++) {
                float64 sumOfGradients = gradientIterator[c];
                float64 sumOfHessians = statisticVector.hessian_diagonal(c);

                // Calculate the score to be predicted for the current label...
                float64 score = sumOfHessians + l2RegularizationWeight_;
                score = score != 0 ? -sumOfGradients / score : 0;
                scoreIterator[c] = score;

                // Calculate the quality score for the current label...
                float64 scorePow = score * score;
                score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
                qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight_ * scorePow);
                overallQualityScore += score;
            }

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
            scoreVector.overallQualityScore = overallQualityScore;
        }

        void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                            DenseScoreVector& scoreVector, int dsysvLwork, float64* dsysvTmpArray1,
                                            int* dsysvTmpArray2, double* dsysvTmpArray3,
                                            float64* dspmvTmpArray) override {
            uint32 numPredictions = scoreVector.getNumElements();
            DenseScoreVector::score_iterator scoreIterator = scoreVector.scores_begin();

            // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
            copyCoefficients(statisticVector, dsysvTmpArray1, numPredictions);
            addRegularizationWeight(dsysvTmpArray1, numPredictions, l2RegularizationWeight_);
            copyOrdinates(statisticVector, scoreIterator, numPredictions);
            lapackPtr_->dsysv(dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, scoreIterator, numPredictions,
                              dsysvLwork);

            // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
            float64 overallQualityScore = blasPtr_->ddot(scoreIterator, statisticVector.gradients_begin(),
                                                         numPredictions);
            blasPtr_->dspmv(statisticVector.hessians_begin(), scoreIterator, dspmvTmpArray, numPredictions);
            overallQualityScore += 0.5 * blasPtr_->ddot(scoreIterator, dspmvTmpArray, numPredictions);

            // Add the L2 regularization term to the overall quality score...
            overallQualityScore += 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);
            scoreVector.overallQualityScore = overallQualityScore;
        }

    public:

        /**
         * @param numPredictions            The number of labels for which the rules may predict
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        RegularizedExampleWiseRuleEvaluation(uint32 numPredictions, float64 l2RegularizationWeight,
                                             std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation(lapackPtr, numPredictions),
              l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr) {

        }

};

static inline void equalWidthBinning(const DenseExampleWiseStatisticVector& statisticVector, float64* coefficients,
                                     float64* ordinates, uint32 numPositiveBins, uint32 numNegativeBins) {
    uint32 n = numPositiveBins + numNegativeBins;

    // Set arrays to zero...
    for (uint32 c = 0; c < n; c++) {
        ordinates[c] = 0;
        uint32 offset = c * n;

        for (uint32 r = 0; r < c + 1; r++) {
            coefficients[offset + r] = 0;
        }
    }

    // Find minimum and maximum gradients...
    uint32 numGradients = statisticVector.getNumElements();
    DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator = statisticVector.gradients_cbegin();

    // TODO Simplify by using std::array
    float64 minPositive = std::numeric_limits<float64>::max();
    float64 maxPositive = 0;
    float64 minNegative = 0;
    float64 maxNegative = std::numeric_limits<float64>::min();

    for (uint32 i = 0; i < numGradients; i++) {
        float64 gradient = gradientIterator[i];

        if (gradient < 0) {
            if (gradient < minNegative) {
                minNegative = gradient;
            }

            if (gradient > maxNegative) {
                maxNegative = gradient;
            }
        } else if (gradient > 0) {
            if (gradient < minPositive) {
                minPositive = gradient;
            }

            if (gradient > maxPositive) {
                maxPositive = gradient;
            }
        }
    }

    float64 spanPerPositiveBin = maxPositive > 0 ? (maxPositive - minPositive) / numPositiveBins : 0;
    float64 spanPerNegativeBin = minNegative < 0 ? (maxNegative - minNegative) / numNegativeBins : 0;

    for (uint32 i = 0; i < numGradients; i++) {
        float64 gradient = gradientIterator[i];
        uint32 binIndex;

        if (gradient < 0) {
            // Gradient belongs to a negative bin...
            binIndex = floor((gradient - minNegative) / spanPerNegativeBin);

            if (binIndex >= numNegativeBins) {
                binIndex = numNegativeBins - 1;
            }
        } else if (gradient > 0) {
            // Gradient belongs to a positive bin...
            binIndex = floor((gradient - minPositive) / spanPerPositiveBin);

            if (binIndex >= numPositiveBins) {
                binIndex = numPositiveBins - 1;
            }

            binIndex += numNegativeBins;
        }

        ordinates[binIndex] -= gradient;
        // TODO Add Hessians to bin
    }
}

/**
 * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
 * Hessians that have been calculated according to a loss function that is applied example wise using L2 regularization.
 * The labels are assigned to bins based on the corresponding gradients.
 */
class BinningExampleWiseRuleEvaluation : public AbstractExampleWiseRuleEvaluation {

    private:

        float64 l2RegularizationWeight_;

        uint32 numPositiveBins_;

        uint32 numNegativeBins_;

        std::shared_ptr<Blas> blasPtr_;

    protected:

        void calculateLabelWisePrediction(const DenseExampleWiseStatisticVector& statisticVector,
                                          LabelWiseEvaluatedPrediction& prediction) override {
            // TODO
        }

        void calculateExampleWisePrediction(DenseExampleWiseStatisticVector& statisticVector,
                                            DenseScoreVector& scoreVector, int dsysvLwork, float64* dsysvTmpArray1,
                                            int* dsysvTmpArray2, double* dsysvTmpArray3,
                                            float64* dspmvTmpArray) override {
            uint32 numPredictions = scoreVector.getNumElements();
            EvaluatedPrediction::score_iterator scoreIterator = scoreVector.scores_begin();

            // Apply equal-width binning...
            equalWidthBinning(statisticVector, dsysvTmpArray1, scoreIterator, numPositiveBins_, numNegativeBins_);
            // TODO
        }

    public:

        /**
         * @param numPositiveBins           The number of bins to be used for labels that should be predicted
         *                                  positively. Must be at least 1
         * @param numNegativeBins           The number of bins to be used for labels that should be predicted
         *                                  negatively. Must be at least 1
         * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
         *                                  scores to be predicted by rules
         * @param blasPtr                   A shared pointer to an object of type `Blas` that allows to execute
         *                                  different BLAS routines
         * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
         *                                  different LAPACK routines
         */
        BinningExampleWiseRuleEvaluation(uint32 numPositiveBins, uint32 numNegativeBins, float64 l2RegularizationWeight,
                                         std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
            : AbstractExampleWiseRuleEvaluation(lapackPtr, numPositiveBins + numNegativeBins),
              l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
              numNegativeBins_(numNegativeBins), blasPtr_(blasPtr) {

        }

};

RegularizedExampleWiseRuleEvaluationFactoryImpl::RegularizedExampleWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight, std::shared_ptr<Blas> blasPtr, std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactoryImpl::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_,
                                                                  blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> RegularizedExampleWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<RegularizedExampleWiseRuleEvaluation>(indexVector.getNumElements(), l2RegularizationWeight_,
                                                                  blasPtr_, lapackPtr_);
}

BinningExampleWiseRuleEvaluationFactoryImpl::BinningExampleWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight, uint32 numPositiveBins, uint32 numNegativeBins, std::shared_ptr<Blas> blasPtr,
        std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), numPositiveBins_(numPositiveBins),
      numNegativeBins_(numNegativeBins), blasPtr_(blasPtr), lapackPtr_(lapackPtr) {

}

std::unique_ptr<IExampleWiseRuleEvaluation> BinningExampleWiseRuleEvaluationFactoryImpl::create(
        const FullIndexVector& indexVector) const {
    return std::make_unique<BinningExampleWiseRuleEvaluation>(numPositiveBins_, numNegativeBins_,
                                                              l2RegularizationWeight_, blasPtr_, lapackPtr_);
}

std::unique_ptr<IExampleWiseRuleEvaluation> BinningExampleWiseRuleEvaluationFactoryImpl::create(
        const PartialIndexVector& indexVector) const {
    return std::make_unique<BinningExampleWiseRuleEvaluation>(numPositiveBins_, numNegativeBins_,
                                                              l2RegularizationWeight_, blasPtr_, lapackPtr_);
}
