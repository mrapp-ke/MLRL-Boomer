/**
 * Provides classes that implement different strategies for finding the heads of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt)
 */
#pragma once

#include "../../common/cpp/head_refinement.h"
#include "lift_functions.h"


namespace seco {

    /**
     * Allows to find the best head that predicts for one or several labels depending on a lift function.
     */
    class PartialHeadRefinementImpl : virtual public IHeadRefinement {

        private:

            std::shared_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param liftFunctionPtr A shared pointer to an object of type `ILiftFunction` that should affect the
             *                        quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinementImpl(std::shared_ptr<ILiftFunction> liftFunctionPtr);

            bool findHead(const PredictionCandidate* bestHead, std::unique_ptr<PredictionCandidate>& headPtr,
                          const uint32* labelIndices, IStatisticsSubset& statisticsSubset, bool uncovered,
                          bool accumulated) const override;

            const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                           bool accumulated) const override;

    };

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

            std::unique_ptr<IHeadRefinement> create() const override;

    };

}
