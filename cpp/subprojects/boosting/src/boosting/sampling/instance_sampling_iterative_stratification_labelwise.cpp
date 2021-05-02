#include "boosting/sampling/instance_sampling_iterative_stratification_labelwise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"

#include <algorithm>

namespace boosting{

typedef  std::map<std::vector<uint32>, std::vector<float32>> mapDesiredSamples;

    IterativeStratificationLabelWise::IterativeStratificationLabelWise(float32 sampleSize):sampleSize_(sampleSize){}

    std::unique_ptr<IWeightVector> IterativeStratificationLabelWise::subSample(const BiPartition& partition, RNG& rng,
                                                             const IRandomAccessLabelMatrix& labelMatrix,
                                                             const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng);
    }

    std::unique_ptr<IWeightVector> IterativeStratificationLabelWise::subSample(const SinglePartition& partition, RNG& rng,
                                                             const IRandomAccessLabelMatrix& labelMatrix,
                                                             const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng);
    }

    mapExamples IterativeStratificationLabelWise::findExamplesPerLabel(const IRandomAccessLabelMatrix& labelMatrixPtr)const{
        uint32 numExamples = labelMatrixPtr.getNumRows();
        uint32 numLabels = labelMatrixPtr.getNumCols();

        mapExamples examplesPerLabel;
        for(uint32 exampleIndex=0; exampleIndex<numExamples; exampleIndex++){
                for(uint32 labelIndex=0; labelIndex<numLabels; labelIndex++){
                    uint8 label = labelMatrixPtr.getValue(exampleIndex, labelIndex);
                    if (label == 0){
                        continue;
                    }else{
                        if(examplesPerLabel.find({labelIndex}) != examplesPerLabel.end()){
                            examplesPerLabel[{labelIndex}].push_back(exampleIndex);
                        }else{
                            examplesPerLabel.insert({{labelIndex}, {exampleIndex}});
                        }
                    }
                }
        }
        return examplesPerLabel;
    }

    uint32Vector IterativeStratificationLabelWise::getNextLabel(mapExamples& examplesPerLabel)const{
        uint32Vector currentChoose = {};
        uint32 minValue = 0;

        for (auto it = examplesPerLabel.begin(); it != examplesPerLabel.end(); it++){
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

    uint32 IterativeStratificationLabelWise::getNextSubset(float32Vector& desiredSamplesPerLabel,
                                                           uint32Vector& desiredSamplesPerSet)const{
        uint32Vector M;
        float32 maxValue = *std::max_element(desiredSamplesPerLabel.begin(), desiredSamplesPerLabel.end());

        for(uint32 i=0; i<desiredSamplesPerLabel.size(); i++){
            if(desiredSamplesPerLabel[i]==maxValue){
                M.push_back(i);
            }
        }

        if(M.size() == 1){
            return M[0];
        }else{
            uint32Vector MM;
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

    std::unique_ptr<IWeightVector> IterativeStratificationLabelWise::subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                                                RNG& rng)const{
       uint32 numExamples = labelMatrixPtr.getNumRows();
       uint32 numSamples = (uint32) (numExamples * sampleSize_);
       uint32Vector desiredSamplesPerSet = {numSamples, numExamples - numSamples};

       mapExamples examplesPerLabel = findExamplesPerLabel(labelMatrixPtr);
       mapDesiredSamples desiredSamplesPerLabelPerSet;

       for (auto it = examplesPerLabel.begin(); it != examplesPerLabel.end(); it++){
           float32Vector desiredSamples = {(float32)(sampleSize_ * it->second.size()), (float32)((1-sampleSize_) * it->second.size())};
           desiredSamplesPerLabelPerSet.insert({it->first, desiredSamples});
       }

       std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
       DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

       uint32Vector l = getNextLabel(examplesPerLabel);
       while (!l.empty()){
           while(examplesPerLabel[l].size() > 0){
               uint32 randomIndex = rng.random(0, examplesPerLabel[l].size());
               auto indexIter = examplesPerLabel[l].begin();
               advance(indexIter, randomIndex);
               uint32 currentSample = *indexIter;
               uint32 m = getNextSubset(desiredSamplesPerLabelPerSet[l], desiredSamplesPerSet);

               if(m==0){
                   weightIterator[currentSample] += 1;
                }

               for (auto labelIter=examplesPerLabel.begin(); labelIter!=examplesPerLabel.end();labelIter++){
                   if((std::find(labelIter->second.begin(), labelIter->second.end(), currentSample) != labelIter->second.end())){
                      desiredSamplesPerLabelPerSet[labelIter->first][m] -= 1;
                   }
                    examplesPerLabel[labelIter->first].remove(currentSample);
               }
               desiredSamplesPerSet[m] -= 1;
           }
           l = getNextLabel(examplesPerLabel);
       }
       return weightVectorPtr;
    }
}