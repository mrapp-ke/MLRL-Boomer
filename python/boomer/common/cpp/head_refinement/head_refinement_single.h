/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "head_refinement_factory.h"


/**
 * Allows to create instances of the class `SingleLabelHeadRefinementImpl`.
 */
class SingleLabelHeadRefinementFactory : public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create(const FullIndexVector& labelIndices) const override;

        std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const override;

};
