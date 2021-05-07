#include "boosting/sampling/instance_sampling_gradient_based_labelwise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/statistics/statistics.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include <cmath>

typedef  std::map<std::vector<uint32>, std::vector<float32>> mapDesiredSamples;

namespace boosting{

    bool compareLabelwise(const sample& first, const sample& second){
        if (first.gradient > second.gradient)
            return true;
        else
            return false;
    }

    GradientBasedLabelWise::GradientBasedLabelWise(float32 sampleSizeTop, float32 sampleSizeRandom):
            sampleSizeTop_(sampleSizeTop), sampleSizeRandom_(sampleSizeRandom){}

    std::unique_ptr<IWeightVector> GradientBasedLabelWise::subSample(const BiPartition& partition, RNG& rng,
                                                                    const IRandomAccessLabelMatrix& labelMatrix,
                                                                    const IStatistics& statistics) const{
        return subSample_(labelMatrix, rng, statistics);
    }


    std::unique_ptr<IWeightVector> GradientBasedLabelWise::subSample(const SinglePartition& partition, RNG& rng,
                                                                     const IRandomAccessLabelMatrix& labelMatrix,
                                                                     const IStatistics& statistics)const{
        return subSample_(labelMatrix, rng, statistics);
    }


    const mapExamplesGradients GradientBasedLabelWise::visit(const IRandomAccessLabelMatrix& labelMatrix,
                                              const DenseLabelWiseStatisticMatrix& statisticMatrix)const{
        return findExamplesPerLabel(labelMatrix, statisticMatrix);
    }

    const mapExamplesGradients GradientBasedLabelWise::visit(const IRandomAccessLabelMatrix& labelMatrix,
                                              const DenseExampleWiseStatisticMatrix& statisticMatrix)const{
        return findExamplesPerLabel(labelMatrix, statisticMatrix);
    }

    template<typename DenseStatisticMatrix>
    const mapExamplesGradients GradientBasedLabelWise::findExamplesPerLabel(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                             const DenseStatisticMatrix& statisticMatrix)const{

        uint32 numExamples = labelMatrixPtr.getNumRows();
        uint32 numLabels = labelMatrixPtr.getNumCols();

        mapExamplesGradients examplesPerLabel;
        for(uint32 exampleIndex=0; exampleIndex<numExamples; exampleIndex++){
            for(uint32 labelIndex=0; labelIndex<numLabels; labelIndex++){
                uint8 label = labelMatrixPtr.getValue(exampleIndex, labelIndex);

                auto gradient_iter = statisticMatrix.gradients_row_cbegin(exampleIndex);
                float64 gradient = std::abs(gradient_iter[labelIndex]);

                if (label == 0){
                    continue;
                }else{
                    if(examplesPerLabel.find({labelIndex}) != examplesPerLabel.end()){
                        examplesPerLabel[{labelIndex}].push_back({exampleIndex, gradient});
                    }else{
                        examplesPerLabel.insert({{labelIndex}, {{exampleIndex, gradient}}});
                    }
                }
            }
        }
        return examplesPerLabel;
    }

    std::unique_ptr<IWeightVector> GradientBasedLabelWise::subSample_(const IRandomAccessLabelMatrix& labelMatrixPtr,
                                                                      RNG& rng, const IStatistics& statistics)const{

       uint32 numExamples = labelMatrixPtr.getNumRows();
       uint32 numSamplesTop = (uint32) (numExamples * sampleSizeTop_);
       uint32 numSamplesRandom = (uint32) (numExamples * sampleSizeRandom_);

       uint32Vector desiredSamplesPerSet = {numSamplesTop, numExamples - numSamplesTop};

       mapExamplesGradients examplesPerLabel = dynamic_cast<const IBoostingStatistics*>(&statistics)->visit(labelMatrixPtr, *this);

       for(auto it=examplesPerLabel.begin(); it!=examplesPerLabel.end(); it++){
            sort(it->second.begin(), it->second.end(),compareLabelwise);
       }

       mapDesiredSamples desiredSamplesPerLabelPerSet;
       for(auto it=examplesPerLabel.begin(); it!=examplesPerLabel.end(); it++){
           uint32 numSamplesPerLabel = it->second.size();
           float32Vector desiredSamplesTop = {(float32)(sampleSizeTop_ * numSamplesPerLabel), (float32)((1-sampleSizeTop_) * numSamplesPerLabel)};
           desiredSamplesPerLabelPerSet.insert({it->first, desiredSamplesTop});
       }

       uint32 numSamples = numSamplesTop + numSamplesRandom;
       std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
       DenseWeightVector::iterator weightIterator = weightVectorPtr->begin();

       std::vector<bool> usedExamples(numExamples, false);
       uint32Vector l = getNextLabel(examplesPerLabel);
       uint32 labelused = 1;
       while (!l.empty()){
          while(examplesPerLabel[l].size() > 0){
               uint32 m = getNextSubset(desiredSamplesPerLabelPerSet[l], desiredSamplesPerSet);
               if (m==0){
                   uint32 currentSample = examplesPerLabel[l].begin()->index;

                   if(!usedExamples[currentSample]){
                       usedExamples[currentSample] = true;
                       weightIterator[currentSample] = 1;
                   }
                   examplesPerLabel[l].erase(examplesPerLabel[l].begin());
               }else if(m==1){
                   examplesPerLabel[l].erase(examplesPerLabel[l].end()-1);
               }
               desiredSamplesPerLabelPerSet[l][m] -= 1;
               desiredSamplesPerSet[m] -= 1;
          }
          labelused +=1;
          l = getNextLabel(examplesPerLabel);
       }
       if (sampleSizeRandom_!=0.0){
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

    uint32Vector GradientBasedLabelWise::getNextLabel(mapExamplesGradients examplesPerLabel)const{
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

    uint32 GradientBasedLabelWise::getNextSubset(float32Vector& desiredSamplesPerLabel,
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
}