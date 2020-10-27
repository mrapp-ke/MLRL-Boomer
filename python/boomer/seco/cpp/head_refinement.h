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
     * Allows to find the best head that predicts for one or several labels depending on a lift function.
     *
     * @tparam T The type of the vector that provides access to the indices of the labels that are considered when
     *           searching for the best head
     */
    template<class T>
    class PartialHeadRefinementImpl : virtual public IHeadRefinement {

        private:

            const T& labelIndices_;

            std::shared_ptr<ILiftFunction> liftFunctionPtr_;

            std::unique_ptr<PartialPrediction> headPtr_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels that should be considered when searching for the best head
             * @param liftFunctionPtr   A shared pointer to an object of type `ILiftFunction` that should affect the
             *                          quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinementImpl(const T& labelIndices, std::shared_ptr<ILiftFunction> liftFunctionPtr);

            const PredictionCandidate* findHead(const PredictionCandidate* bestHead,
                                                IStatisticsSubset& statisticsSubset, bool uncovered,
                                                bool accumulated) override;

            std::unique_ptr<PredictionCandidate> pollHead() override;

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

            std::unique_ptr<IHeadRefinement> create(const RangeIndexVector& labelIndices) const override;

            std::unique_ptr<IHeadRefinement> create(const DenseIndexVector& labelIndices) const override;

    };

}
