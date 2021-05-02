#include "boosting/sampling/instance_sampling_iterative_stratification_labelset.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"
#include <vector>
#include <map>
#include <list>
#include <algorithm>

namespace boosting{

typedef  std::map<std::vector<uint32>, std::vector<float32>> mapDesiredSamples;

    IterativeStratificationLabelSet::IterativeStratificationLabelSet(float32 sampleSize):sampleSize_(sampleSize){}

    std::unique_ptr<IWeightVector> IterativeStratificationLabelSet::subSample(const BiPartition& partition, RNG& rng,
                                                             const IRandomAccessLabelMatrix& labelMatrix,
                                                             const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng);
    }

    std::unique_ptr<IWeightVector> IterativeStratificationLabelSet::subSample(const SinglePartition& partition, RNG& rng,
                                                             const IRandomAccessLabelMatrix& labelMatrix,
                                                             const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng);
    }

    mapExamples IterativeStratificationLabelSet::findExamplesPerLabelset(const IRandomAccessLabelMatrix& labelMatrixPtr)const{
        uint32 numExamples = labelMatrixPtr.getNumRows();
        uint32 numLabels = labelMatrixPtr.getNumCols();

        mapExamples examplesPerLabelset;

        for(uint32 exampleIndex=0; exampleIndex<numExamples; exampleIndex++){
            uint32Vector labelSet;
            for(uint32 labelIndex=0; labelIndex<numLabels; labelIndex++){
                labelSet.push_back(labelMatrixPtr.getValue(exampleIndex, labelIndex));
            }
            if(examplesPerLabelset.find({labelSet}) != examplesPerLabelset.end()){
                examplesPerLabelset[{labelSet}].push_back(exampleIndex);
            }else{
                examplesPerLabelset.insert({{labelSet}, {exampleIndex}});
            }
        }
        return examplesPerLabelset;
    }

    uint32Vector IterativeStratificationLabelSet::getNextLabelset(mapExamples& examplesPerLabel)const{
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

    uint32 IterativeStratificationLabelSet::getNextSubset(float32Vector& desiredSamplesPerLabelset,
                                                          uint32Vector& desiredSamplesPerSet)const{
        uint32Vector M;
        float32 maxValue = *std::max_element(desiredSamplesPerLabelset.begin(), desiredSamplesPerLabelset.end());

        for(uint32 i=0; i<desiredSamplesPerLabelset.size(); i++){
            if(desiredSamplesPerLabelset[i]==maxValue){
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

    std::unique_ptr<IWeightVector> IterativeStratificationLabelSet::subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                                               RNG& rng)const{
       uint32 numExamples = labelMatrixPtr.getNumRows();
       uint32 numSamples = (uint32) (numExamples * sampleSize_);
       uint32Vector desiredSamplesPerSet = {numSamples, numExamples - numSamples};

       mapExamples examplesPerLabelset = findExamplesPerLabelset(labelMatrixPtr);

       mapDesiredSamples desiredSamplesPerLabelsetPerSet;

       for (auto it = examplesPerLabelset.begin(); it != examplesPerLabelset.end(); it++){
           float32Vector desiredSamples = {(float32)(sampleSize_ * it->second.size()), (float32)((1-sampleSize_) * it->second.size())};
           desiredSamplesPerLabelsetPerSet.insert({it->first, desiredSamples});
       }

       std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
       DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

       uint32Vector l = getNextLabelset(examplesPerLabelset);
       while (!l.empty()){
           while(examplesPerLabelset[l].size() > 0){
               uint32 randomIndex = rng.random(0, examplesPerLabelset[l].size());
               auto indexIter = examplesPerLabelset[l].begin();
               advance(indexIter, randomIndex);
               uint32 currentSample = *indexIter;
               uint32 m = getNextSubset(desiredSamplesPerLabelsetPerSet[l], desiredSamplesPerSet);

               if(m==0){
                   weightIterator[currentSample] += 1;
               }

               for (auto labelIter=examplesPerLabelset.begin(); labelIter!=examplesPerLabelset.end(); labelIter++){
                   if((std::find(labelIter->second.begin(), labelIter->second.end(), currentSample) != labelIter->second.end())){
                      desiredSamplesPerLabelsetPerSet[labelIter->first][m] -= 1;
                   }
                    examplesPerLabelset[labelIter->first].remove(currentSample);
               }
               desiredSamplesPerSet[m] -= 1;
           }
           l = getNextLabelset(examplesPerLabelset);
       }
       return weightVectorPtr;
    }
}