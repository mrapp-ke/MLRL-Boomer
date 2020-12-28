/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "body.h"
#include "condition_list.h"


/**
 * A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions, and = or !=
 * for nominal conditions, respectively.
 */
class ConjunctiveBody : public IBody {

    private:

        uint32 numLeq_;

        uint32* leqFeatureIndices_;

        float32* leqThresholds_;

        uint32 numGr_;

        uint32* grFeatureIndices_;

        float32* grThresholds_;

        uint32 numEq_;

        uint32* eqFeatureIndices_;

        float32* eqThresholds_;

        uint32 numNeq_;

        uint32* neqFeatureIndices_;

        float32* neqThresholds_;

    public:

        /**
         * @param conditionList A reference to an object of type `ConditionList` that provides access to the conditions,
         *                      the body should contain
         */
        ConjunctiveBody(const ConditionList& conditionList);

        ~ConjunctiveBody();

        bool covers(CContiguousFeatureMatrix::const_iterator begin,
                    CContiguousFeatureMatrix::const_iterator end) const override;

        bool covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                    CsrFeatureMatrix::index_const_iterator indicesEnd,
                    CsrFeatureMatrix::value_const_iterator valuesBegin,
                    CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

};
