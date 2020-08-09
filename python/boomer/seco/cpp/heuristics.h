/**
 * Implements different heuristics for assessing the quality of single- or multi-label rules based on confusion
 * matrices. Given the elements of a confusion matrix, such a heuristic calculates a quality score in [0, 1].
 *
 * All heuristics must be implemented as loss functions, i.e., rules with a smaller quality score are considered better
 * than those with a large quality score.
 *
 * All heuristics must treat positive and negative labels equally, i.e., if the ground truth and a rule's predictions
 * would be inverted, the resulting quality scores must be the same as before.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"


namespace seco {

    /**
     * The number of elements in a confusion matrix.
     */
    const intp NUM_CONFUSION_MATRIX_ELEMENTS = 4;

    /**
     * An enum that specified all positive elements of a confusion matrix.
     */
    enum ConfusionMatrixElement : intp {
        IN = 0,
        IP = 1,
        RN = 2,
        RP = 3
    };

    /**
     * Returns the confusion matrix element, a label corresponds to, depending on the ground truth an a prediction.
     *
     * @param trueLabel         The true label according to the ground truth
     * @param predictedLabel    The predicted label
     * @return                  The confusion matrix element
     */
    static inline ConfusionMatrixElement getConfusionMatrixElement(uint8 trueLabel, uint8 predictedLabel) {
        if (trueLabel) {
            return predictedLabel ? RP : RN;
        } else {
            return predictedLabel ? IP : IN;
        }
    }

    /**
     * An abstract base class for all heuristics that allows to calculate quality scores based on the elements of
     * confusion matrices.
     */
    class AbstractHeuristic {

        public:

            virtual ~AbstractHeuristic();

            /**
             * Calculates and returns a quality score in [0, 1] given the elements of a confusion matrix. All elements
             * must be equal to or greater than 0. If a rule does not cover any elements, i.e., if
             * `CIN + CIP + CRN + CRP == 0`, the worst possible quality score 1 must be returned.
             *
             * According to the notation in http://www.ke.tu-darmstadt.de/bibtex/publications/show/3201, a confusion
             * matrix consists of 8 elements, namely CIN, CIP, CRN, CRP, UIN, UIP, URN and URP. The individual symbols
             * used in this notation have the following meaning:
             *
             * - The first symbol denotes whether the corresponding labels are covered (C) or uncovered (U) by the rule.
             * - The second symbol denotes relevant (R) or irrelevant (I) labels according to the ground truth.
             * - The third symbol denotes labels for which the prediction in the rule's head is positive (P) or negative
             *   (N).
             *
             * This results in the terminology given in the following table:
             *
             *      | ground-   |           |
             *      | truth     | predicted |
             * -----------|-----------|-----------|-----
             *  covered   |         0 |         0 | CIN
             *            |         0 |         1 | CIP
             *            |         1 |         0 | CRN
             *            |         1 |         1 | CRP
             * -----------|-----------|-----------|-----
             *  uncovered |         0 |         0 | UIN
             *            |         0 |         1 | UIP
             *            |         1 |         0 | URN
             *            |         1 |         1 | URP
             *
             * Real numbers may be used for the individual elements, if different weights are assigned to the
             * corresponding labels.
             *
             * @param cin   The number of covered (C) labels that are irrelevant (I) according to the ground truth and
             *              for which the prediction in the rule's head is negative (N)
             * @param cip   The number of covered (C) labels that are irrelevant (I) according to the ground truth and
             *              for which the prediction in the rule's head is negative (N)
             * @param crn   The number of covered (C) labels that are relevant (R) according to the ground truth and for
             *              which the prediction in the rule's head is negative (N)
             * @param crp   The number of covered (C) labels that are relevant (R) according to the ground truth and for
             *              which the prediction in the rule's head is positive (P)
             * @param uin   The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and
             *              for which the prediction in the rule's head is negative (N)
             * @param uip   The number of uncovered (U) labels that are irrelevant (I) according to the ground truth and
             *              for which the prediction in the rule's head is positive (P)
             * @param urn   The number of uncovered (U) labels that are relevant (R) according to the ground truth and
             *              for which the prediction in the rule's head is negative (N)
             * @param urp   The number of uncovered (U) labels that are relevant (R) according to the ground truth and
             *              for which the prediction in the rule's head is positive (P)
             * @return      The quality score that has been calculated
             */
            virtual float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                    float64 uip, float64 urn, float64 urp);

    };

    /**
     * A heuristic that measures the fraction of incorrectly predicted labels among all covered labels.
     */
    class PrecisionImpl : public AbstractHeuristic {

        public:

            ~PrecisionImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

    /**
     * A heuristic that measures the fraction of uncovered labels among all labels for which the rule's prediction is
     * (or would be) correct, i.e., for which the ground truth is equal to the rule's prediction.
     */
    class RecallImpl : public AbstractHeuristic {

        public:

            ~RecallImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

    /**
     * A heuristic that calculates as `1 - wra`, where `wra` corresponds to the weighted relative accuracy metric.
     */
    class WRAImpl : public AbstractHeuristic {

        public:

            ~WRAImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

    /**
     * A heuristic that measures the fraction of incorrectly predicted labels among all labels.
     */
    class HammingLossImpl : public AbstractHeuristic {

        public:

            ~HammingLossImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

    /**
     * A heuristic that calculates as the (weighted) harmonic mean between the heuristics `PrecisionImpl` and
     * `RecallImpl`, where the parameter `beta` allows to trade-off between both heuristics. If `beta = 1`, both
     * heuristics are weighed equally. If `beta = 0`, this heuristic is equivalent to the heuristic `PrecisionImpl`. As
     * `beta` approaches infinity, this heuristic becomes equivalent to the heuristic `RecallImpl`.
     */
    class FMeasureImpl : public AbstractHeuristic {

        private:

            /**
             * The value of the beta-parameter.
             */
            float64 beta_;

        public:

            /**
             * Creates a new heuristic that calculates as the (weighted) harmonic mean between the heuristics
             * `PrecisionImpl` and `RecallImpl`.
             *
             * @param beta The value of the beta-parameter. Must be at least 0
             */
            FMeasureImpl(float64 beta);

            ~FMeasureImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

    /**
     * A heuristic that allows to trade off between the heuristics `PrecisionImpl` and `WRAFunction`. The `m`-parameter
     * allows to control the trade-off between both heuristics. If `m = 0`, this heuristic is equivalent to the
     * heuristic `PrecisionImpl`. As `m` approaches infinity, the isometrics of this heuristic become equivalent to
     * those of the heuristic `WRAFunction`.
     */
    class MEstimateImpl : public AbstractHeuristic {

        private:

            /**
             * The value of the m-parameter.
             */
            float64 m_;

        public:

            /**
             * Creates a new heuristic that allows to trade off between the heuristics `PrecisionImpl` and
             * `WRAFunction`.
             *
             * @param m The value of the m-parameter. Must be at least 0
             */
            MEstimateImpl(float64 beta);

            ~MEstimateImpl();

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) override;

    };

}
