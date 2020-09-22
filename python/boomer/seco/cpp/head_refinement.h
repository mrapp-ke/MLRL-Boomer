/**
 * Provides classes that implement different strategies for finding the heads of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt)
 */
#pragma once

#include "../../common/cpp/head_refinement.h"
#include "lift_functions.h"
#include <memory>


namespace seco {

    class PartialHeadRefinementImpl : virtual public IHeadRefinement {

        private:

            std::shared_ptr<AbstractLiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param liftFunctionPtr A shared pointer to an object of type `AbstractLiftFunction` that should affect
             *                        the quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinementImpl(std::shared_ptr<AbstractLiftFunction> liftFunctionPtr);

            PredictionCandidate* findHead(PredictionCandidate* bestHead, PredictionCandidate* recyclableHead,
                                          const uint32* labelIndices, AbstractStatisticsSubset* statisticsSubset,
                                          bool uncovered, bool accumulated) override;

            PredictionCandidate* calculatePrediction(AbstractStatisticsSubset* statisticsSubset, bool uncovered,
                                                     bool accumulated) override;

    };

}
