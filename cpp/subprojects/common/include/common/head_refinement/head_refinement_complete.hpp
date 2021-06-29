/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/head_refinement_factory.hpp"


/**
 * Allows to create instances of the class `CompleteHeadRefinement`.
 */
class CompleteHeadRefinementFactory final : public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create(const CompleteIndexVector& labelIndices) const override;

        std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const override;

};
