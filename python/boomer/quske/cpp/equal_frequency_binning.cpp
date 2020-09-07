#include "equal_frequency_binning.h"

intp EqualFrequencyBinning::numBins_;

void EqualFrequencyBinningImpl::createBins(intp numBins_){
    intp length = originalMatrix->size;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins_){
        //TODO: inform observer we have no new matrix
        return;
    }
    /*  Simple Python Implementation as example
     *  def equal_frequency(arr, k):    #arr -> originalMatrix, k -> numBins_
     *      length = len(arr)           #length of originalMatrix
     *      n = int(length/k)   #Number of elements per bin
     *      result = []                 #has to be allocated - size numBins_
     *      for i in range(0, k):       #for(intp i = 0; i < numBins_; i++)
     *          tmp = []                #has not to be allocated - size number of features
     *          for j in range(i * n, (i + 1) * n): #for(intp j = i * n; j < (i + 1) * n; j++)
     *              if j >= length:     #break out if j >= originalMatrix' length
     *                  break           #
     *              tmp = tmp + [arr[j]]#add currently handled row to the bin
     *          result.append(tmp)      #write tmp to i-st spot in result array
     *      return result               #return pointer
     */
    intp n = length/numBins_; //number of elements per bin
    //Initializing result array
    indexedFloatArray *results = (indexedFloatArray*)malloc(sizeof(indexedFloatArray));
    results->size = numBins_;
    float *resultData = (float*)malloc(numThresholds * sizeof(float)); //TODO: Change type from float to float array
    results->data = resultData;
    numFeatures = 7; //TODO: Replace "7" with the number of features
    for(intp i = 0; i < numBins_; i++){
        float tmp[numFeatures] {0};
        for(intp j = i * n; j < (i + 1) * n; j++){
            if(j >= length){
                break;
            }
            for(intp k = 0; k < numFeatures; k++){ //adding all feature values
                tmp[k] = tmp[k] + originalMatrix->data[j][k];
            }
        }
        result->data[i] = tmp;
    }
    //TODO: Inform observer that a new matrix is available
}