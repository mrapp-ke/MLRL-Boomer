#include "stratified_sampling.hpp"
#include <algorithm>

IterativeStratification::IterativeStratification(uint32 numSets):
    numSets(numSets){
        for (uint32 i=0; i<numSets; i++){distributions.push_back(1.0/float64(numSets));}
    }

IterativeStratification::IterativeStratification(uint32 numSets, std::vector<float64> distributions):
    numSets(numSets), distributions(distributions){}

uint32 IterativeStratification::getCombinationWithFewestExamples(std::map<uint32, std::list<uint32>>& samplesPerLabel){
    int currentChoose = -1;
    uint32 minValue = 0;
    std::map<uint32, std::list<uint32>>::iterator it;
    for (it = samplesPerLabel.begin(); it != samplesPerLabel.end(); it++){
        uint32 size = it->second.size();
        if (size == 0){
            continue;
        }
        if (currentChoose == -1 || (size != 0 && size < minValue)){
            currentChoose = it->first;
            minValue = size;
        }
    }
    return currentChoose;
}

uint32 IterativeStratification::getSubsetWithMaxDesiredSamples
(std::vector<float64>& desiredSamplesPerLabelPerSet, std::vector<float64>& desiredSamplesPerSet){
    std::vector<uint32> M;
    float64 maxValue = *std::max_element(desiredSamplesPerLabelPerSet.begin(), desiredSamplesPerLabelPerSet.end());

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

std::map<uint32, std::list<uint32>> IterativeStratification::stratify(CContiguousLabelMatrix& labelMatrixPtr){
   uint32 numExamples = labelMatrixPtr.getNumRows();
   uint32 numLabels = labelMatrixPtr.getNumCols();

   // calculate the desired number of examples at each subset
   std::vector<float64> desiredSamplesPerSet = distributions;
   for_each(desiredSamplesPerSet.begin(),desiredSamplesPerSet.end(), [k=float64(numExamples)](float64 &c){ c *= k;});

   // calculate the desired number of examples of each label at each subset
   std::map<uint32, std::list<uint32>> examplesPerLabel;

   for(uint32 exampleIndex=0; exampleIndex<numExamples; exampleIndex++){
       for(uint32 labelIndex=0; labelIndex<numLabels; labelIndex++){
           uint8 label = labelMatrixPtr.getValue(exampleIndex, labelIndex);
           if (label == 0){
               continue;
           }else{
               if(examplesPerLabel.find(labelIndex) != examplesPerLabel.end()){
                   examplesPerLabel[labelIndex].push_back(exampleIndex);
               }else{
                   examplesPerLabel.insert({labelIndex, {exampleIndex}});
               }
            }
        }
    }

    std::map<uint32, std::vector<float64>> desiredSamplesPerLabelPerSet;
    std::map<uint32, std::list<uint32>>::iterator it;

    for (it = examplesPerLabel.begin(); it != examplesPerLabel.end(); it++){
        std::vector<float64> desiredSamples = distributions;
        std::for_each(desiredSamples.begin(), desiredSamples.end(), [k=float64(it->second.size())](float64 &c){ c *= k; });
        desiredSamplesPerLabelPerSet.insert({it->first, desiredSamples});
    }

    // Find the label with the fewest (but al least one) remaining examples, breaking ties randomly
    std::map<uint32, std::list<uint32>> subsets;
    for (uint32 setIndex=0; setIndex<numSets; setIndex++){
        subsets.insert({setIndex, {}});
    }

    int l = getCombinationWithFewestExamples(examplesPerLabel);
    while (l != -1){
        while(examplesPerLabel[l].size() > 0){
            uint32 currentSample = *examplesPerLabel[l].begin();
        //std::list<uint32>::iterator samplesIter;
        //for (samplesIter = examplesPerLabel[l].begin(); samplesIter != examplesPerLabel[l].end(); samplesIter++){
             uint32 m = getSubsetWithMaxDesiredSamples(desiredSamplesPerLabelPerSet[l], desiredSamplesPerSet);
             subsets[m].push_back(currentSample);

             std::map<uint32, std::list<uint32>>::iterator labelIter;
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
    return subsets;
}