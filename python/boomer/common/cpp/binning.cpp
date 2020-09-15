#include "binning.h"

AbstractBinning::~AbstractBinning(){

}

void AbstractBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){

}


void BinningObserver::onBinUpdate(intp binIndex, IndexedFloat32* indexedValue){

};


EqualFrequencyBinning::EqualFrequencyBinning(intp numBins){
    numBins_ = numBins;
}

void EqualFrequencyBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){
    intp length = indexedArray->numElements;
    //Mandatory block skipping the process, if the condition is already satisfied
    if(length <= numBins_){
        //TODO: inform observer that no approximation is necessary
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
    IndexedFloat32Array *results = (IndexedFloat32Array*)malloc(sizeof(IndexedFloat32Array));
    results->numElements = numBins_;
    IndexedFloat32 *resultData = (IndexedFloat32*)malloc(numBins_ * sizeof(IndexedFloat32));
    results->data = resultData;
    //looping over bins
    for(intp i = 0; i < numBins_; i++){
        float tmp;
        tmp = 0;
        //looping over feature list between two bins
        for(intp j = i * n; j < ((i + 1) * n); j++){
            //if we would break out of bounds we have to break out of the loop
            if(j >= length){
                break;
            }
            //here we aggregate the values
            tmp = tmp + indexedArray->data[j].value;
        }
        results->data[i].value = tmp;
        tmp = 0;
    }
    //TODO: Inform observer that a new matrix is available
}


EqualWidthBinning::EqualWidthBinning(intp numBins){
    numBins_ = numBins;
}

void EqualWidthBinning::createBins(IndexedFloat32Array* indexedArray, BinningObserver* observer){
    intp length = indexedArray->numElements;
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
     //defining minimal and maximum values
     float min = indexedArray->data[0].value;
     intp bound_min = floor(min);
     float max = indexedArray->data[indexedArray->numElements-1].value;
     //w stands for width and determines the span of values for a bin
     intp w = intp(ceil((max - min)/numBins_));
      //Initializing result array
     IndexedFloat32Array *results = (IndexedFloat32Array*)malloc(sizeof(IndexedFloat32Array));
     results->numElements = numBins_;
     IndexedFloat32 *resultData = (IndexedFloat32*)malloc(numBins_ * sizeof(IndexedFloat32));
     results->data = resultData;
     //defining the boundaries of bins
     intp boundaries[numBins_ + 1] {0};
     for(intp i = 0; i < numBins_ + 1; i++){
        boundaries[i] = bound_min + w * i;
     }
     //looping over bins
     for(intp i = 0; i < numBins_; i++){
        float tmp = 0;
        //looping over the list and adding every element in bin i
        for(intp j = 0; j < length; j++){
            if(boundaries[i]<= j && j < boundaries[i + 1]){
                tmp = tmp + indexedArray->data[j].value;
            }
        }
        results->data[i].value = tmp;
     }
     //TODO: Inform observer that a new matrix is available
}

