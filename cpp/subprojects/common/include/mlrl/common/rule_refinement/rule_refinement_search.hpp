/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"
#include "mlrl/common/input/feature_vector_nominal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/input/feature_vector_ordinal.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_fixed.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"
#include "mlrl/common/statistics/statistics_weighted_immutable.hpp"

/**
 * Allows to conduct a search for finding the best refinement of an existing rule that can be created from a given
 * feature vector.
 */
class RuleRefinementSearch final {
    public:

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given numerical
         * feature vector.
         *
         * @param featureVector A reference to an object of type `NumericalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinement
         */
        void searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                          const MissingFeatureVector& missingFeatureVector,
                                          const IImmutableWeightedStatistics& statistics,
                                          SingleRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given numerical
         * feature vector.
         *
         * @param featureVector A reference to an object of type `NumericalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `MultiRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinements
         */
        void searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                          const MissingFeatureVector& missingFeatureVector,
                                          const IImmutableWeightedStatistics& statistics,
                                          FixedRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given nominal
         * feature vector.
         *
         * @param featureVector A reference to an object of type `NominalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinement
         */
        void searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        const IImmutableWeightedStatistics& statistics,
                                        SingleRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given nominal
         * feature vector.
         *
         * @param featureVector A reference to an object of type `NominalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `MultiRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinements
         */
        void searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        const IImmutableWeightedStatistics& statistics,
                                        FixedRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given binary feature
         * vector.
         *
         * @param featureVector A reference to an object of type `BinaryFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinement
         */
        void searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       const IImmutableWeightedStatistics& statistics,
                                       SingleRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given binary feature
         * vector.
         *
         * @param featureVector A reference to an object of type `BinaryFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `MultiRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinements
         */
        void searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       const IImmutableWeightedStatistics& statistics,
                                       FixedRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given ordinal
         * feature vector.
         *
         * @param featureVector A reference to an object of type `OrdinalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `SingleRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinement
         */
        void searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        const IImmutableWeightedStatistics& statistics,
                                        SingleRefinementComparator& comparator, uint32 minCoverage) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a given ordinal
         * feature vector.
         *
         * @param featureVector A reference to an object of type `OrdinalFeatureVector`, the refinements should be
         *                      created from
         * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that provides access to
         *                      weighted statistics about the labels of the training examples, which should serve as the
         *                      basis for evaluating the quality of potential refinements
         * @param comparator    A reference to an object of type `MultiRefinementComparator` that should be used for
         *                      comparing potential refinements
         * @param minCoverage   The minimum number of examples that must be covered by the refinements
         */
        void searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        const IImmutableWeightedStatistics& statistics,
                                        FixedRefinementComparator& comparator, uint32 minCoverage) const;
};