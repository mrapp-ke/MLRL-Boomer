#include "boosting/sampling/instance_sampling_gradient_based_labelset.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"
#include "boosting/statistics/statistics_example_wise_common.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include <cmath>

typedef  std::map<std::vector<uint32>, std::vector<float32>> mapDesiredSamples;

namespace boosting{
    bool compareLabelset(const sample& first, const sample& second){
        if (first.gradient > second.gradient)
            return true;
        else
            return false;
    }
    GradientBasedLabelSet::GradientBasedLabelSet(float32 sampleSizeTop, float32 sampleSizeRandom):
            sampleSizeTop_(sampleSizeTop), sampleSizeRandom_(sampleSizeRandom){}

    std::unique_ptr<IWeightVector> GradientBasedLabelSet::subSample(const BiPartition& partition, RNG& rng,
                                                                    const IRandomAccessLabelMatrix& labelMatrix,
                                                                    const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng, statistics);
    }


    std::unique_ptr<IWeightVector> GradientBasedLabelSet::subSample(const SinglePartition& partition, RNG& rng,
                                                                    const IRandomAccessLabelMatrix& labelMatrix,
                                                                    const IStatistics& statistics)const{
        return subSample_(labelMatrix, rng, statistics);
    }


    const mapExamplesGradients GradientBasedLabelSet::visit(const IRandomAccessLabelMatrix& labelMatrix,
                                                            const DenseLabelWiseStatisticMatrix& statisticMatrix)const{
        return findExamplesPerLabelset(labelMatrix, statisticMatrix);
    }

    const mapExamplesGradients GradientBasedLabelSet::visit(const IRandomAccessLabelMatrix& labelMatrix,
                                                            const DenseExampleWiseStatisticMatrix& statisticMatrix)const{
        return findExamplesPerLabelset(labelMatrix, statisticMatrix);
    }

    template<typename DenseStatisticMatrix>
    const mapExamplesGradients GradientBasedLabelSet::findExamplesPerLabelset(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                               const DenseStatisticMatrix& statisticMatrix)const{
        uint32 numExamples = labelMatrixPtr.getNumRows();
        uint32 numLabels = labelMatrixPtr.getNumCols();

        mapExamplesGradients examplesPerLabelset;

        for(uint32 exampleIndex=0; exampleIndex<numExamples; exampleIndex++){

            uint32Vector labelSet;
            for(uint32 labelIndex=0; labelIndex<numLabels; labelIndex++){
                labelSet.push_back(labelMatrixPtr.getValue(exampleIndex, labelIndex));
            }

            float64 gradient = 0;
            auto gradient_iter = statisticMatrix.gradients_row_cbegin(exampleIndex);
            while(gradient_iter!=statisticMatrix.gradients_row_cend(exampleIndex)){
                gradient += std::abs(*gradient_iter);
                ++gradient_iter;
            }

            if(examplesPerLabelset.find({labelSet}) != examplesPerLabelset.end()){
                examplesPerLabelset[{labelSet}].push_back({exampleIndex, gradient});
            }else{
                examplesPerLabelset.insert({{labelSet}, {{exampleIndex, gradient}}});
            }
        }
        return examplesPerLabelset;
    }

    std::unique_ptr<IWeightVector> GradientBasedLabelSet::subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                                     RNG& rng, const IStatistics& statistics)const{
       uint32 numExamples = labelMatrixPtr.getNumRows();
       uint32 numSamplesTop = (uint32) (numExamples * sampleSizeTop_);
       uint32 numSamplesRandom = (uint32) (numExamples * sampleSizeRandom_);

       uint32Vector desiredSamplesPerSet = {numSamplesTop, numExamples - numSamplesTop};

       mapExamplesGradients examplesPerLabelset = dynamic_cast<const IBoostingStatistics*>(&statistics)->visit(labelMatrixPtr, *this);

       for(auto it=examplesPerLabelset.begin(); it!=examplesPerLabelset.end(); it++){
            sort(it->second.begin(), it->second.end(),compareLabelset);
       }

       mapDesiredSamples desiredSamplesPerLabelsetPerSet;
       for(auto it=examplesPerLabelset.begin(); it!=examplesPerLabelset.end(); it++){
           uint32 numSamplesPerLabelset = it->second.size();
           float32Vector desiredSamplesTop = {(float32)(sampleSizeTop_ * numSamplesPerLabelset), (float32)((1-sampleSizeTop_) * numSamplesPerLabelset)};
           desiredSamplesPerLabelsetPerSet.insert({it->first, desiredSamplesTop});
       }

       uint32 numSamples = numSamplesTop + numSamplesRandom;
       std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
       DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

       std::vector<bool> usedExamples(numExamples, false);

       uint32Vector l = getNextLabelset(examplesPerLabelset);
       while (!l.empty()){
          while(examplesPerLabelset[l].size() > 0){
               uint32 m = getNextSubset(desiredSamplesPerLabelsetPerSet[l], desiredSamplesPerSet);

               if (m==0){
                   uint32 currentSample = examplesPerLabelset[l].begin()->index;

                   if(!usedExamples[currentSample]){
                       usedExamples[currentSample] = true;
                       weightIterator[currentSample] = 1;
                   }
                   examplesPerLabelset[l].erase(examplesPerLabelset[l].begin());
               }else if(m==1){
                   examplesPerLabelset[l].erase(examplesPerLabelset[l].end()-1);
               }
               desiredSamplesPerLabelsetPerSet[l][m] -= 1;
               desiredSamplesPerSet[m] -= 1;
          }
          l = getNextLabelset(examplesPerLabelset);
       }

        if(sampleSizeRandom_!=0.0){
           float32 fuct = (1-sampleSizeTop_)/sampleSizeRandom_;
           uint32 randomExampleIndex = rng.random(0, numExamples);
           while(numSamplesRandom>0 && !usedExamples[randomExampleIndex]){
               --numSamplesRandom;
               usedExamples[randomExampleIndex] = true;
               weightIterator[randomExampleIndex] = fuct;
               randomExampleIndex = rng.random(0, numExamples);
           }
        }
       return weightVectorPtr;
    }

    uint32Vector GradientBasedLabelSet::getNextLabelset(mapExamplesGradients examplesPerLabelset)const{
        uint32Vector currentChoose = {};
        uint32 minValue = 0;

        for (auto it = examplesPerLabelset.begin(); it != examplesPerLabelset.end(); it++){
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

    uint32 GradientBasedLabelSet::getNextSubset(float32Vector& desiredSamplesPerLabelset,
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
}