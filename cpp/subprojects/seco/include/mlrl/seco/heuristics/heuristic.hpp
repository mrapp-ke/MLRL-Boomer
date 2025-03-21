/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

namespace seco {

    /**
     * Defines an interface for all heuristics that allows to calculate numerical scores that assess the quality of
     * rules, based on the elements of confusion matrices. Given the elements of a confusion matrix, such a heuristic
     * calculates a numerical score in [0, 1].
     *
     * All heuristics must be implemented as gain metrics, i.e., rules with a greater numerical score are considered
     * better than those with a smaller numerical score.
     *
     * All heuristics must treat positive and negative labels equally, i.e., if the ground truth and a rule's
     * predictions would be inverted, the resulting numerical scores must be the same as before.
     */
    class IHeuristic {
        public:

            virtual ~IHeuristic() {}

            /**
             * Calculates and returns a numerical score in [0, 1] given the elements of a confusion matrix. All elements
             * must be equal to or greater than 0. If a rule does not cover any elements, i.e., if
             * `CIN + CIP + CRN + CRP == 0`, the worst possible quality 0 must be returned.
             *
             * According to the notation in https://ke-tud.github.io/bibtex/attachments/single/432, a confusion matrix
             * consists of 8 elements, namely CIN, CIP, CRN, CRP, UIN, UIP, URN and URP. The individual symbols used in
             * this notation have the following meaning:
             *
             * - The first symbol denotes whether the corresponding labels are covered (C) or uncovered (U) by the rule.
             * - The second symbol denotes relevant (R) or irrelevant (I) labels according to the ground truth.
             * - The third symbol denotes labels for which the prediction of a rule is positive (P) or negative (N).
             *
             * This results in the terminology given in the following table:
             *
             *            | ground-   |           |
             *            | truth     | predicted |
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
             * Real numbers may be used for the individual elements, if real-valued weights are assigned to the
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
             * @return      The quality that has been calculated
             */
            virtual float32 evaluateConfusionMatrix(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin,
                                                    float32 uip, float32 urn, float32 urp) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IHeuristic`.
     */
    class IHeuristicFactory {
        public:

            virtual ~IHeuristicFactory() {}

            /**
             * Creates and returns a new object of type `IHeuristic`.
             *
             * @return An unique pointer to an object of type `IHeuristic` that has been created
             */
            virtual std::unique_ptr<IHeuristic> create() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a heuristic.
     */
    class IHeuristicConfig {
        public:

            virtual ~IHeuristicConfig() {}

            /**
             * Creates and returns a new object of type `IHeuristicFactory` according to the specified configuration.
             *
             * @return An unique pointer to an object of type `IHeuristicFactory` that has been created
             */
            virtual std::unique_ptr<IHeuristicFactory> createHeuristicFactory() const = 0;
    };

}
