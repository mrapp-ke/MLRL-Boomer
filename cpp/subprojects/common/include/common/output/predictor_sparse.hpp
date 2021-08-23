/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/matrix_lil.hpp"
#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/input/label_vector_set.hpp"
#include "common/model/rule_model.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to make predictions for given query examples using an existing
 * rule-based model and write them into a sparse matrix.
 *
 * @tparam T The type of the values that are stored by the prediction matrix
 */
template<typename T>
class ISparsePredictor {

    public:

        virtual ~ISparsePredictor() { };

        /**
         * Obtains predictions for all examples in a C-contiguous matrix, using a specific rule-based model, and stores
         * them in a sparse matrix in the list of lists (LIL) format.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param model             A reference to an object of type `RuleModel` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `LilMatrix` that stores the predictions
         */
        virtual std::unique_ptr<LilMatrix<T>> predict(const CContiguousFeatureMatrix& featureMatrix,
                                                      const RuleModel& model,
                                                      const LabelVectorSet* labelVectors) const = 0;

        /**
         * Obtains predictions for all examples in a sparse CSR matrix, using a specific rule-based model, and stores
         * them in a sparse matrix in the list of lists (LIL) format.
         *
         * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature values
         *                          of the examples
         * @param model             A reference to an object of type `RuleModel` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `LilMatrix` that stores the predictions
         */
        virtual std::unique_ptr<LilMatrix<T>> predict(const CsrFeatureMatrix& featureMatrix, const RuleModel& model,
                                                      const LabelVectorSet* labelVectors) const = 0;

};
