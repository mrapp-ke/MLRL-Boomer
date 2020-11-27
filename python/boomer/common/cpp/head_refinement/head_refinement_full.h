/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "head_refinement_factory.h"


/**
 * Allows to create instances of the class `FullHeadRefinementImpl`.
 */
class FullHeadRefinementFactory : public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create() const override;

};
