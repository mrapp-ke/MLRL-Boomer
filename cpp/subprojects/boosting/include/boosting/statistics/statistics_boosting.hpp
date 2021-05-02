#pragma once
#include "common/input/label_matrix.hpp"
#include <vector>
#include <map>
#include <list>

struct sample{
    uint32 index;
    float64 gradient;
};


typedef  std::map<std::vector<uint32>, std::vector<sample>> mapExamplesGradients;

namespace boosting{

    class DenseExampleWiseStatisticMatrix;
    class DenseLabelWiseStatisticMatrix;

    class IVisitor{
        public:
            virtual ~IVisitor(){};
            virtual const mapExamplesGradients visit(const IRandomAccessLabelMatrix& labelMatrix,
                                                const DenseExampleWiseStatisticMatrix& statisticMatrix)const=0;
            virtual const mapExamplesGradients visit(const IRandomAccessLabelMatrix& labelMatrix,
                                                const DenseLabelWiseStatisticMatrix& statisticMatrix)const=0;
    };


    class IBoostingStatistics{
        public:
            virtual ~IBoostingStatistics(){};
            virtual const mapExamplesGradients visit(const IRandomAccessLabelMatrix& labelMatrix,
                                                     const IVisitor& visitor)const=0;
    };
}