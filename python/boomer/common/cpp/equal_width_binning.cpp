#include "equal_width_binning.h"
#include <math.h>

intp EqualWidthBinning::numBins_;

EqualWidthBinning::EqualWidthBinning(intp numBins){
    numBins_ = numBins;
}

void EqualWidthBinningImpl::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer);{
    intp length = originalMatrix->size;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins_){
        //TODO: inform observer that no approximation is necessary
        return;
    }
    /*  Simple Python Implementation as example
     *  def equal_width(arr, k):    #arr -> originalMatrix, k -> numBins_
     *                              #should define length as originalMatrix size
     *      min_arr = min(arr)      #originalMatrix[0][currentFeature]
     *      w = int(ceil((max(arr)-min_arr)/k))   #Width of the interval
     *      result = []             #has to be allocated - size numBins_
     *      boundaries = []         #has not to allocated - size numBins_+1
     *      for i in range(0, k+1): #for(intp i = 0; i < numBins_ + 1; i++)
     *          boundaries = boundaries + [min_arr + w * i]
     *      for i in range(0, k):   #for(intp i = 0; i < numBins_; i++)
     *          tmp = []            #has not to be allocated - size number of features
     *          for j in arr:       #for(intp j = 0; j < length; j++)
     *              if boundaries[i] <= j < boundaries[i + 1]:  #j = originalMatrix[i][currentFeature]
     *                  tmp += [j]  #every element from tmp + every element from j
     *          result += [tmp]     #write tmp to the i-st element of the result array
     *      return result           #return pointer to the result array
     */
    /*  float min = originalMatrix[0][currentFeatureIndex];
     *  intp bound_min = floor(min);
     *  float max = originalMatrix[originalMatrix->size][currentFeatureIndex];
     *  intp w = int(ceil((max - min)/numBins_))
        //Initializing result array
     *  indexedFloatArray *results = (indexedFloatArray*)malloc(sizeof(indexedFloatArray));
     *  results->size = numBins_;
     *  float *resultData = (float*)malloc(numThresholds * sizeof(float)); //TODO: Change type from float to float array?
     *  results->data = resultData;
     *  numFeatures = 7; //TODO: Replace "7" with the number of features
     *  intp boundaries[numBins_ + 1] {0};
     *  for(intp i = 0; i < numBins_ + 1; i++){
     *      boundaries[i] = bound_min + w * i;
     *  }
     *  for(intp i = 0; i < numBins_; i++){
     *      float tmp[numFeatures] {0};
     *      for(intp j = 0; j < length; j++){
     *          if(boundaries[i]<= j && j < boundaries[i + 1]){
     *              for(intp k = 0; k < numFeatures; k++){ //adding all feature values
     *                  tmp[k] = tmp[k] + originalMatrix->data[j][k];
     *              }
     *          }
     *      }
     *      result->data[i] = tmp;
     *  }
     * //TODO: Inform observer that a new matrix is available
     */
}