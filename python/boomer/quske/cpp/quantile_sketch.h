/**
 * Provides the base implementation of the quantile sketch class
 */

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"

class QuantileSketch{

    private:

        // INDEXEDARRAY* QuantileSketch::approximatedStatisticsObjects;

        intp QuantileSketch::currentFeatureIterator;

        bool QuantileSketch::filled;

        AbstractStatistics* equalWidthQS(intp thresholdTarget, AbstractStatistics allThresholds, intp orgLength);

        //matrix* aggregate(matrix *gradientMatrix, intp step);

    public:

        AbstractStatistics* approximate(intp methodID, intp thresholdTarget, AbstractStatistics allThresholds, intp orgLength);

        QuantileSketch();

        ~QuantileSketch();

};