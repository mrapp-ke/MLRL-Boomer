/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_refinement/rule_refinement.hpp"
#include "common/rule_refinement/rule_refinement_callback.hpp"
#include "common/binning/bin_vector.hpp"
#include "common/head_refinement/head_refinement.hpp"


/**
 * Allows to find the best refinements of existing rules, which result from adding a new condition that correspond to a
 * certain feature. The thresholds that may be used by the new condition result from the boundaries between the bins
 * that have been created using a binning method.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the refined rule is
 *           allowed to predict
 */
template<class T>
class ApproximateRuleRefinement final : public IRuleRefinement {

    private:

        std::unique_ptr<IHeadRefinement> headRefinementPtr_;

        const T& labelIndices_;

        uint32 featureIndex_;

        bool nominal_;

        std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>> callbackPtr_;

        std::unique_ptr<Refinement> refinementPtr_;

    public:

        /**
         * @param headRefinementPtr An unique pointer to an object of type `IHeadRefinement` that should be used to find
         *                          the head of refined rules
         * @param labelIndices      A reference to an object of template type `T` that provides access to the indices of
         *                          the labels for which the refined rule is allowed to predict
         * @param featureIndex      The index of the feature, the new condition corresponds to
         * @param nominal           True, if the feature at index `featureIndex` is nominal, false otherwise
         * @param callbackPtr       An unique pointer to an object of type `IRuleRefinementCallback` that allows to
         *                          retrieve the bins for a certain feature
         */
        ApproximateRuleRefinement(std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices,
                                  uint32 featureIndex, bool nominal,
                                  std::unique_ptr<IRuleRefinementCallback<BinVector, DenseVector<uint32>>> callbackPtr);

        void findRefinement(const AbstractEvaluatedPrediction* currentHead) override;

        std::unique_ptr<Refinement> pollRefinement() override;

};
