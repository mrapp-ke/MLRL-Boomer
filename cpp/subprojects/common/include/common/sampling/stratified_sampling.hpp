#ifndef STRATIFICATION_HPP_
#define STRATIFICATION_HPP_
#include <vector>
#include <map>
#include <list>
#include "types.hpp"
#include "label_matrix_c_contiguous.hpp"

class IterativeStratification {
    private:
        uint32 numSets;
        std::vector<float64> distributions;
        uint32 getCombinationWithFewestExamples(std::map<uint32, std::list<uint32>>& samplesPerLabel);
        uint32 getSubsetWithMaxDesiredSamples(std::vector<float64>& desiredSamplesPerSet, std::vector<float64>& desiredSamplesPerLabelPerSet);

    public:
        IterativeStratification(uint32 numSets);
        IterativeStratification(uint32 numSets, std::vector<float64> distributions);

        ~IterativeStratification(){};

        std::map<uint32, std::list<uint32>> stratify(CContiguousLabelMatrix& labelMatrixPtr);
};
#endif