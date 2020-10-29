/**
 * Provides classes that implement different strategies for finding the heads of rules.
 *
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt)
 */
#pragma once

#include "../../common/cpp/head_refinement.h"
#include "lift_functions.h"


namespace seco {

    /**
     * Allows to create instances of the class `PartialHeadRefinementImpl`.
     */
    class PartialHeadRefinementFactoryImpl : virtual public IHeadRefinementFactory {

        private:

            std::shared_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param liftFunctionPtr A shared pointer to an object of type `ILiftFunction` that should affect the
             *                        quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinementFactoryImpl(std::shared_ptr<ILiftFunction> liftFunctionPtr);

            std::unique_ptr<IHeadRefinement> create(const FullIndexVector& labelIndices) const override;

            std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const override;

    };

}
