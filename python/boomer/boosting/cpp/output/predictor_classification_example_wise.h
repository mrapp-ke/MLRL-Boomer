/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/output/predictor.h"
#include "../../../common/cpp/input/label_vector.h"
#include <unordered_set>
#include <functional>


namespace boosting {

    /**
     * Allows to predict the labels of given query examples using an existing rule-based model that has been learned
     * using a boosting algorithm.
     *
     * For prediction, the scores that are provided by the individual rules, are summed up. For each query example, the
     * aggregated score vector is then compared to   known label sets in order to obtain a distance measure. The label
     * set that is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictor : public IPredictor<uint8> {

        private:

            /**
             * Allows to compute hashes for objects of type `LabelVector`.
             */
            struct HashFunction {

                inline std::size_t operator()(const std::unique_ptr<LabelVector>& v) const {
                    std::size_t hash = (std::size_t) v->getNumElements();

                    for (auto it = v->indices_cbegin(); it != v->indices_cend(); it++) {
                        hash ^= *it + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                    }

                    return hash;
                }

            };

            /**
             * Allows to check whether two objects of type `LabelVector` are equal.
             */
            struct EqualsFunction {

                inline bool operator()(const std::unique_ptr<LabelVector>& lhs,
                                       const std::unique_ptr<LabelVector>& rhs) const {
                    if (lhs->getNumElements() != rhs->getNumElements()) {
                        return false;
                    }

                    auto it1 = lhs->indices_cbegin();

                    for (auto it2 = rhs->indices_cbegin(); it2 != rhs->indices_cend(); it2++) {
                        if (*it1 != *it2) {
                            return false;
                        }

                        it1++;
                    }

                    return true;
                }

            };

            std::unordered_set<std::unique_ptr<LabelVector>, HashFunction, EqualsFunction> labelVectors_;

        public:

            typedef std::function<void(const LabelVector&)> LabelVectorVisitor;

            /**
             * Adds a known label vector that may be predicted for individual query examples.
             *
             * @param labelVectorPtr An unique pointer to an object of type `LabelVector`
             */
            void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr);

            /**
             * Invokes the given visitor function for each unique label vector that has been provided via the function `addLabelVector`.
             *
             * @param visitor The visitor function for handling objects of the type `LabelVector`
             */
            void visit(LabelVectorVisitor visitor) const;

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<uint8>& predictionMatrix,
                         const RuleModel& model) const override;

    };

}
