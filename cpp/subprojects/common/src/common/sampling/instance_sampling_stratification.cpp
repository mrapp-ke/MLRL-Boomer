#include "common/sampling/instance_sampling_stratification.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include <vector>
#include <map>
#include <list>
#include <algorithm>

std::map<std::vector<uint32>, std::list<uint32>> findExamplesPerLabel(const IRandomAccessLabelMatrix& labelMatrixPtr, bool strategy ){
    uint32 numExamples_ = labelMatrixPtr.getNumRows();
    uint32 numLabels_ = labelMatrixPtr.getNumCols();

    std::map<std::vector<uint32>, std::list<uint32>> examplesPerLabel_;
    if(strategy){
        for(uint32 exampleIndex=0; exampleIndex<numExamples_; exampleIndex++){
            std::vector<uint32> labelSet;
            for(uint32 labelIndex=0; labelIndex<numLabels_; labelIndex++){
                labelSet.push_back(labelMatrixPtr.getValue(exampleIndex, labelIndex));
            }
            if(examplesPerLabel_.find({labelSet}) != examplesPerLabel_.end()){
                examplesPerLabel_[{labelSet}].push_back(exampleIndex);
            }else{
                examplesPerLabel_.insert({{labelSet}, {exampleIndex}});
            }
        }
    }else{
        for(uint32 exampleIndex=0; exampleIndex<numExamples_; exampleIndex++){
            for(uint32 labelIndex=0; labelIndex<numLabels_; labelIndex++){
                uint8 label = labelMatrixPtr.getValue(exampleIndex, labelIndex);
                if (label == 0){
                    continue;
                }else{
                    if(examplesPerLabel_.find({labelIndex}) != examplesPerLabel_.end()){
                        examplesPerLabel_[{labelIndex}].push_back(exampleIndex);
                    }else{
                        examplesPerLabel_.insert({{labelIndex}, {exampleIndex}});
                    }
                }
            }
        }
    }

    return examplesPerLabel_;
}

std::vector<uint32> getCombinationWithFewestExamples(std::map<std::vector<uint32>, std::list<uint32>>& samplesPerLabel){
    std::vector<uint32> currentChoose = {};
    uint32 minValue = 0;
    std::map<std::vector<uint32>, std::list<uint32>>::iterator it;
    for (it = samplesPerLabel.begin(); it != samplesPerLabel.end(); it++){
        uint32 size = it->second.size();
        if (size == 0){
            continue;
        }
        if (currentChoose.empty() || (size != 0 && size < minValue)){
            currentChoose = it->first;
            minValue = size;
        }
    }
    return currentChoose;
}

uint32 getSubsetWithMaxDesiredSamples(std::vector<float32>& desiredSamplesPerLabelPerSet, std::vector<uint32>& desiredSamplesPerSet){
    std::vector<uint32> M;
    float32 maxValue = *std::max_element(desiredSamplesPerLabelPerSet.begin(), desiredSamplesPerLabelPerSet.end());

    for(uint32 i=0; i<desiredSamplesPerLabelPerSet.size(); i++){
        if(desiredSamplesPerLabelPerSet[i]==maxValue){
            M.push_back(i);
        }
    }

    if(M.size() == 1){
        return M[0];
    }else{
        std::vector<uint32> MM;
        maxValue = *std::max_element(desiredSamplesPerSet.begin(), desiredSamplesPerSet.end());
        for(uint32 i=0; i<desiredSamplesPerSet.size(); i++){
                if (desiredSamplesPerSet[i]==maxValue){
                    MM.push_back(i);
                }
        }
        if (M.size()==1){
            return M[0];
        }else{
            return MM[rand() % MM.size()];
        }
    }
}

std::unique_ptr<IWeightVector> subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr, uint32 sampleSize_, RNG& rng){
   uint32 numExamples = labelMatrixPtr.getNumRows();

   // calculate the desired number of examples at each subset
   uint32 numSamples = (uint32) (numExamples * sampleSize_);
   std::vector<uint32> desiredSamplesPerSet = {numSamples, numExamples - numSamples};

   // Find the examples of each label in the initial set
   std::map<std::vector<uint32>, std::list<uint32>> examplesPerLabel = findExamplesPerLabel(labelMatrixPtr, false);

   // calculate the desired number of examples of each label at each subset
   std::map<std::vector<uint32>, std::vector<float32>> desiredSamplesPerLabelPerSet;
   std::map<std::vector<uint32>, std::list<uint32>>::iterator it;

   for (it = examplesPerLabel.begin(); it != examplesPerLabel.end(); it++){
       std::vector<float32> desiredSamples = {(float32)(sampleSize_ * it->second.size()), (float32)((1-sampleSize_) * it->second.size())};
       desiredSamplesPerLabelPerSet.insert({it->first, desiredSamples});
   }

   // Find the label with the fewest (but al least one) remaining examples, breaking ties randomly
   std::map<uint32, std::list<uint32>> subsets;
   for (uint32 setIndex=0; setIndex<2; setIndex++){
       subsets.insert({setIndex, {}});
   }

   std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
   DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

   std::vector<uint32> l = getCombinationWithFewestExamples(examplesPerLabel);
   while (!l.empty()){
       while(examplesPerLabel[l].size() > 0){
           uint32 randomIndex = rng.random(0, examplesPerLabel[l].size());
           auto indexIter = examplesPerLabel[l].begin();
           advance(indexIter, randomIndex);
           uint32 currentSample = *indexIter;
           uint32 m = getSubsetWithMaxDesiredSamples(desiredSamplesPerLabelPerSet[l], desiredSamplesPerSet);
           subsets[m].push_back(currentSample);

           if(m==0){
               weightIterator[currentSample] += 1;
            }

           std::map<std::vector<uint32>, std::list<uint32>>::iterator labelIter;
           for (labelIter=examplesPerLabel.begin(); labelIter!=examplesPerLabel.end();labelIter++){
               if((std::find(labelIter->second.begin(), labelIter->second.end(), currentSample) != labelIter->second.end())){
                  desiredSamplesPerLabelPerSet[labelIter->first][m] -= 1;
               }

                examplesPerLabel[labelIter->first].remove(currentSample);
           }
           desiredSamplesPerSet[m] -= 1;
       }
       l = getCombinationWithFewestExamples(examplesPerLabel);
   }

   return weightVectorPtr;
}

Stratification::Stratification(float32 sampleSize):sampleSize_(sampleSize){}


std::unique_ptr<IWeightVector> Stratification::subSample(const BiPartition& partition, RNG& rng,
                                                         const IRandomAccessLabelMatrix& labelMatrix) const{
    return subSample_(labelMatrix, sampleSize_, rng);
}


std::unique_ptr<IWeightVector> Stratification::subSample(const SinglePartition& partition, RNG& rng,
                                                         const IRandomAccessLabelMatrix& labelMatrix) const{
    return subSample_(labelMatrix, sampleSize_, rng);
}