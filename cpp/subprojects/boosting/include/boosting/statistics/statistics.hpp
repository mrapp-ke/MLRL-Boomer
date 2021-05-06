/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics.hpp"
#include <functional>


namespace boosting {

    // Forward declarations
    class DenseLabelWiseStatisticView;
    class DenseExampleWiseStatisticView;

    /**
     * Defines an interface for all classes that provide access to statistics, which serve as the basis for learning a
     * new gradient boosted rule or refining an existing one.
     */
    class IBoostingStatistics : virtual public IStatistics {

        public:

            virtual ~IBoostingStatistics() { };

            /**
             * A visitor function for handling objects of the type `DenseLabelWiseStatisticView`.
             */
            typedef std::function<void(std::unique_ptr<DenseLabelWiseStatisticView>&)> DenseLabelWiseStatisticViewVisitor;

            /**
             * A visitor function for handling objects of the type `DenseExampleWiseStatisticView`.
             */
            typedef std::function<void(std::unique_ptr<DenseExampleWiseStatisticView>&)> DenseExampleWiseStatisticViewVisitor;

            /**
             * Invokes one of the given visitor functions, depending on which one is able to handle the particular type
             * of matrix that is used to store the statistics.
             *
             * @param denseLabelWiseStatisticViewVisitor    The visitor function for handling objects of the type
             *                                              `DenseLabelWiseStatisticView`
             * @param denseExampleWiseStatisticViewVisitor  The visitor function for handling objects of the type
             *                                              `DenseExampleWiseStatisticView`
             */
            virtual void visit(DenseLabelWiseStatisticViewVisitor denseLabelWiseStatisticViewVisitor,
                               DenseExampleWiseStatisticViewVisitor denseExampleWiseStatisticViewVisitor) = 0;

    };

}
