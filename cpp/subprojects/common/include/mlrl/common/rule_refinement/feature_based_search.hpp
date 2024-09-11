/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_binned.hpp"
#include "mlrl/common/input/feature_vector_missing.hpp"
#include "mlrl/common/input/feature_vector_nominal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/input/feature_vector_ordinal.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_fixed.hpp"
#include "mlrl/common/rule_refinement/refinement_comparator_single.hpp"
#include "mlrl/common/statistics/statistics_weighted.hpp"

/**
 * Allows to conduct a search for finding the best refinement of an existing rule that can be created from a given
 * feature vector.
 */
class FeatureBasedSearch final {
    public:

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `NumericalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `NumericalFeatureVector`, the
         *                                      refinements should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                          const MissingFeatureVector& missingFeatureVector,
                                          SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                          const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                          uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `NumericalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `NumericalFeatureVector`, the
         *                                      refinements should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinements
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForNumericalRefinement(const NumericalFeatureVector& featureVector,
                                          const MissingFeatureVector& missingFeatureVector,
                                          FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                          const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                          uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `NominalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `NominalFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                        const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                        uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `NominalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `NominalFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinements
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForNominalRefinement(const NominalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                        const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                        uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `BinaryFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `BinaryFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                       const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                       uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `BinaryFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `BinaryFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinements
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForBinaryRefinement(const BinaryFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                       const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                       uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from an
         * `OrdinalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `OrdinalFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                        const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                        uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from an
         * `OrdinalFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `OrdinalFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinements
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForOrdinalRefinement(const OrdinalFeatureVector& featureVector,
                                        const MissingFeatureVector& missingFeatureVector,
                                        FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                        const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                        uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `BinnedFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `BinnedFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `SingleRefinementComparator` that
         *                                      should be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinement
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForBinnedRefinement(const BinnedFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       SingleRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                       const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                       uint32 minCoverage, Refinement& refinement) const;

        /**
         * Conducts a search for the best refinement of an existing rule that can be created from a
         * `BinnedFeatureVector`.
         *
         * @param featureVector                 A reference to an object of type `BinnedFeatureVector`, the refinements
         *                                      should be created from
         * @param missingFeatureVector          A reference to an object of type `MissingFeatureVector` that provides
         *                                      access to the indices of training examples with missing feature values
         * @param comparator                    A reference to an object of type `MultiRefinementComparator` that should
         *                                      be used for comparing potential refinements
         * @param statistics                    A reference to an object of type `IWeightedStatistics` that provides
         *                                      access to weighted statistics about the quality of predictions for
         *                                      training examples, which should serve as the basis for evaluating the
         *                                      quality of potential refinements
         * @param outputIndices                 A reference to an object of type `IIndexVector` that provides access to
         *                                      the indices of the outputs for which refinements should predict
         * @param numExamplesWithNonZeroWeights The total number of examples with non-zero weights that may be covered
         *                                      by a refinement
         * @param minCoverage                   The minimum number of examples that must be covered by the refinements
         * @param refinement                    A reference to an object of type `Refinement` that should be used for
         *                                      storing the properties of the best refinement that is found
         */
        void searchForBinnedRefinement(const BinnedFeatureVector& featureVector,
                                       const MissingFeatureVector& missingFeatureVector,
                                       FixedRefinementComparator& comparator, const IWeightedStatistics& statistics,
                                       const IIndexVector& outputIndices, uint32 numExamplesWithNonZeroWeights,
                                       uint32 minCoverage, Refinement& refinement) const;
};
