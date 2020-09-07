#include "quantile_sketch.h"

AbstractStatistics* QuantileSketch::equalWidthQS(intp thresholdTarget, AbstractStatistics *allThresholds, intp orgLength)
{
    /**
     *  if(filled){
     *      return approximatedStatisticsObjects[currentFeatureIterator];
     *   }
     */
    /**
     *  matrix *gradientMatrix = AbstractStatistic.matrix;
     *  matrix *newMatrix = aggregate(gradient_matrix);
     *  AbstractStatistics *approximatedStatistics = new AbstractStatistics(newMatrix, orgLength/numThresholds);
     *  ApproximatedStatisticsObjects-> data[currentFeatureIterator] = approximatedStatistics;
     */
    return allThresholds;
};

/**
 *  matrix* QuantileSketch::aggregate(matrix *gradientMatrix, intp thresholdTarget, intp orgLength, intp step)
 *  {
 *      INDEXEDARRAY *results = (INDEXEDARRAY*)malloc(sizeof(INDEXEDARRAY));
 *      results->size = thresholdsTarget;
 *      float *resultData = (float*)malloc(thresholdTarget * sizeof(float));
 *      results->data = resultData;
 *      intp k,limit;
 *      limit = step;
 *      k = 0;
 *      for(intp i = 0; i <= orgLength; i+=step){
 *          for(intp j = 0; j <= limit; j++){
 *              resultData[i][k] = resultDate[i][k] + gradientMatrix[i][j];
 *          }
            k++;
 *          limit += step;
 *      }
 *      return results;
 *  };
 */

AbstractStatistics* QuantileSketch::approximate(intp methodID, intp thresholdTarget, AbstractStatistics *allThresholds, intp orgLength, intp numFeatures)
{
    currentFeatureIterator++;
    if(!filled){
        filled = (currentFeatureIterator >= numFeatures);
    }
    currentFeatureIterator = currentFeatureIterator % numFeatures;
    if(thresholdTarget >= orgLength || methodID < 0){
      return allThresholds;
    }
    switch(methodID){
        case 0: return equalWidthQS(thresholdTarget, allThresholds, orgLength);
        default: return allThresholds;
    }
};

QuantileSketch::QuantileSketch()
{
    currentFeatureIterator = 0;
    filled = false;
};

QuantileSketch::~QuantileSketch()
{

};