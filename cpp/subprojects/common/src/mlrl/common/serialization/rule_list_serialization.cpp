#include "rule_list_serialization.hpp"

#include "mlrl/common/model/body_conjunctive.hpp"
#include "mlrl/common/model/body_empty.hpp"
#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/model/rule_list.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static const char ATTRIBUTE_TYPE[] = "type";
static const char ATTRIBUTE_DEFAULT_RULE_TAKES_PRECEDENCE[] = "default_rule_takes_precedence";
static const char ATTRIBUTE_NUM_USED_RULES[] = "num_used_rules";
static const char ATTRIBUTE_NUM_RULES[] = "num_rules";
static const char ATTRIBUTE_SCORE_TYPE[] = "score_type";
static const char TYPE_RULE_LIST[] = "RuleList";
static const char DATASET_BODY_TYPE[] = "body_type";
static const char GROUP_BODIES[] = "bodies";
static const char DATASET_OFFSETS[] = "offsets";
static const char DATASET_FEATURE_INDICES[] = "feature_indices";
static const char DATASET_THRESHOLDS[] = "thresholds";
static const char GROUP_HEADS[] = "heads";
static const char DATASET_HEAD_TYPE[] = "head_type";
static const char DATASET_SCORE_OFFSETS[] = "score_offsets";
static const char DATASET_SCORES[] = "scores";
static const char DATASET_INDEX_OFFSETS[] = "index_offsets";
static const char DATASET_OUTPUT_INDICES[] = "output_indices";
static const char SCORE_TYPE_UINT8[] = "uint8";
static const char SCORE_TYPE_FLOAT32[] = "float32";
static const char SCORE_TYPE_FLOAT64[] = "float64";

static const char CONDITIONS_NUMERICAL_LEQ[] = "numerical_leq";
static const char CONDITIONS_NUMERICAL_GR[] = "numerical_gr";
static const char CONDITIONS_ORDINAL_LEQ[] = "ordinal_leq";
static const char CONDITIONS_ORDINAL_GR[] = "ordinal_gr";
static const char CONDITIONS_NOMINAL_EQ[] = "nominal_eq";
static const char CONDITIONS_NOMINAL_NEQ[] = "nominal_neq";

/**
 * The value of the dataset `body_type` for rules with an empty body, i.e., for the default rule.
 */
static constexpr uint8 BODY_TYPE_EMPTY = 0;

/**
 * The value of the dataset `body_type` for rules with a conjunctive body.
 */
static constexpr uint8 BODY_TYPE_CONJUNCTIVE = 1;

/**
 * The value of the dataset `head_type` for rules with a complete head.
 */
static constexpr uint8 HEAD_TYPE_COMPLETE = 0;

/**
 * The value of the dataset `head_type` for rules with a partial head.
 */
static constexpr uint8 HEAD_TYPE_PARTIAL = 1;

[[noreturn]] static void throwInvalidRuleModel(const std::string& reason) {
    throw std::runtime_error("Group \"" + std::string(GROUP_RULE_MODEL) + "\" stores an invalid rule model: " + reason);
}

/**
 * Converts the size of a container into an offset of type `uint32`.
 */
static uint32 toOffset(std::size_t size) {
    if (size > std::numeric_limits<uint32>::max()) {
        throw std::runtime_error("Unable to write rule model: The model is too large");
    }

    return static_cast<uint32>(size);
}

/**
 * Checks whether an array of offsets is consistent, i.e., whether it contains `numRules + 1` elements in increasing
 * order, starting at 0 and ending at `numElements`.
 */
static void validateOffsets(const std::vector<uint32>& offsets, std::size_t numRules, std::size_t numElements,
                            const std::string& description) {
    if (offsets.size() != numRules + 1 || offsets[0] != 0 || offsets[numRules] != numElements) {
        throwInvalidRuleModel("The offsets of " + description + " do not match the number of rules or elements");
    }

    for (std::size_t i = 0; i < numRules; i++) {
        if (offsets[i] > offsets[i + 1]) {
            throwInvalidRuleModel("The offsets of " + description + " are not in increasing order");
        }
    }
}

/**
 * Stores the conditions of a specific type that are contained in the bodies of all rules in a columnar format.
 *
 * @tparam Threshold The type of the thresholds that are used by the conditions
 */
template<typename Threshold>
class ConditionColumn final {
    private:

        const char* name_;

        std::vector<uint32> offsets_;

        std::vector<uint32> featureIndices_;

        std::vector<Threshold> thresholds_;

    public:

        explicit ConditionColumn(const char* name) : name_(name), offsets_({0}) {}

        template<typename IndexIterator, typename ThresholdIterator>
        void addConditions(uint32 numConditions, IndexIterator indices, ThresholdIterator thresholds) {
            for (uint32 i = 0; i < numConditions; i++) {
                featureIndices_.push_back(indices[i]);
                thresholds_.push_back(thresholds[i]);
            }
        }

        void finishRule() {
            offsets_.push_back(toOffset(featureIndices_.size()));
        }

        void write(Hdf5Group& bodies) const {
            Hdf5Group group = bodies.createGroup(name_);
            group.writeDataset<uint32>(DATASET_OFFSETS, offsets_.data(), offsets_.size());
            group.writeDataset<uint32>(DATASET_FEATURE_INDICES, featureIndices_.data(), featureIndices_.size());
            group.writeDataset<Threshold>(DATASET_THRESHOLDS, thresholds_.data(), thresholds_.size());
        }

        void read(const Hdf5Group& bodies, std::size_t numRules) {
            Hdf5Group group = bodies.openGroup(name_);
            offsets_ = group.readDataset<uint32>(DATASET_OFFSETS);
            featureIndices_ = group.readDataset<uint32>(DATASET_FEATURE_INDICES);
            thresholds_ = group.readDataset<Threshold>(DATASET_THRESHOLDS);
            std::string description = "the conditions \"" + std::string(name_) + "\"";

            if (featureIndices_.size() != thresholds_.size()) {
                throwInvalidRuleModel("The number of feature indices and thresholds of " + description + " differ");
            }

            validateOffsets(offsets_, numRules, featureIndices_.size(), description);
        }

        uint32 getNumConditions(uint32 ruleIndex) const {
            return offsets_[ruleIndex + 1] - offsets_[ruleIndex];
        }

        template<typename IndexIterator, typename ThresholdIterator>
        void copyConditions(uint32 ruleIndex, IndexIterator indices, ThresholdIterator thresholds) const {
            uint32 start = offsets_[ruleIndex];
            uint32 end = offsets_[ruleIndex + 1];

            for (uint32 i = start; i < end; i++) {
                indices[i - start] = featureIndices_[i];
                thresholds[i - start] = thresholds_[i];
            }
        }
};

/**
 * Stores the conditions that are contained in the bodies of all rules in a columnar format.
 */
class BodyColumns final {
    private:

        std::vector<uint8> bodyTypes_;

        ConditionColumn<float32> numericalLeq_;

        ConditionColumn<float32> numericalGr_;

        ConditionColumn<int32> ordinalLeq_;

        ConditionColumn<int32> ordinalGr_;

        ConditionColumn<int32> nominalEq_;

        ConditionColumn<int32> nominalNeq_;

        void finishRule(uint8 bodyType) {
            bodyTypes_.push_back(bodyType);
            numericalLeq_.finishRule();
            numericalGr_.finishRule();
            ordinalLeq_.finishRule();
            ordinalGr_.finishRule();
            nominalEq_.finishRule();
            nominalNeq_.finishRule();
        }

    public:

        BodyColumns()
            : numericalLeq_(CONDITIONS_NUMERICAL_LEQ), numericalGr_(CONDITIONS_NUMERICAL_GR),
              ordinalLeq_(CONDITIONS_ORDINAL_LEQ), ordinalGr_(CONDITIONS_ORDINAL_GR), nominalEq_(CONDITIONS_NOMINAL_EQ),
              nominalNeq_(CONDITIONS_NOMINAL_NEQ) {}

        const std::vector<uint8>& getBodyTypes() const {
            return bodyTypes_;
        }

        void addEmptyBody(const EmptyBody& body) {
            this->finishRule(BODY_TYPE_EMPTY);
        }

        void addConjunctiveBody(const ConjunctiveBody& body) {
            numericalLeq_.addConditions(body.getNumNumericalLeq(), body.numerical_leq_indices_cbegin(),
                                        body.numerical_leq_thresholds_cbegin());
            numericalGr_.addConditions(body.getNumNumericalGr(), body.numerical_gr_indices_cbegin(),
                                       body.numerical_gr_thresholds_cbegin());
            ordinalLeq_.addConditions(body.getNumOrdinalLeq(), body.ordinal_leq_indices_cbegin(),
                                      body.ordinal_leq_thresholds_cbegin());
            ordinalGr_.addConditions(body.getNumOrdinalGr(), body.ordinal_gr_indices_cbegin(),
                                     body.ordinal_gr_thresholds_cbegin());
            nominalEq_.addConditions(body.getNumNominalEq(), body.nominal_eq_indices_cbegin(),
                                     body.nominal_eq_thresholds_cbegin());
            nominalNeq_.addConditions(body.getNumNominalNeq(), body.nominal_neq_indices_cbegin(),
                                      body.nominal_neq_thresholds_cbegin());
            this->finishRule(BODY_TYPE_CONJUNCTIVE);
        }

        void write(Hdf5Group& group) const {
            group.writeDataset<uint8>(DATASET_BODY_TYPE, bodyTypes_.data(), bodyTypes_.size());
            Hdf5Group bodies = group.createGroup(GROUP_BODIES);
            numericalLeq_.write(bodies);
            numericalGr_.write(bodies);
            ordinalLeq_.write(bodies);
            ordinalGr_.write(bodies);
            nominalEq_.write(bodies);
            nominalNeq_.write(bodies);
        }

        void read(const Hdf5Group& group, std::size_t numRules) {
            bodyTypes_ = group.readDataset<uint8>(DATASET_BODY_TYPE);

            if (bodyTypes_.size() != numRules) {
                throwInvalidRuleModel("The number of body types does not match the number of rules");
            }

            Hdf5Group bodies = group.openGroup(GROUP_BODIES);
            numericalLeq_.read(bodies, numRules);
            numericalGr_.read(bodies, numRules);
            ordinalLeq_.read(bodies, numRules);
            ordinalGr_.read(bodies, numRules);
            nominalEq_.read(bodies, numRules);
            nominalNeq_.read(bodies, numRules);
        }

        /**
         * Creates and returns the conjunctive body of a specific rule.
         */
        std::unique_ptr<ConjunctiveBody> createConjunctiveBody(uint32 ruleIndex) const {
            std::unique_ptr<ConjunctiveBody> bodyPtr = std::make_unique<ConjunctiveBody>(
              numericalLeq_.getNumConditions(ruleIndex), numericalGr_.getNumConditions(ruleIndex),
              ordinalLeq_.getNumConditions(ruleIndex), ordinalGr_.getNumConditions(ruleIndex),
              nominalEq_.getNumConditions(ruleIndex), nominalNeq_.getNumConditions(ruleIndex));
            numericalLeq_.copyConditions(ruleIndex, bodyPtr->numerical_leq_indices_begin(),
                                         bodyPtr->numerical_leq_thresholds_begin());
            numericalGr_.copyConditions(ruleIndex, bodyPtr->numerical_gr_indices_begin(),
                                        bodyPtr->numerical_gr_thresholds_begin());
            ordinalLeq_.copyConditions(ruleIndex, bodyPtr->ordinal_leq_indices_begin(),
                                       bodyPtr->ordinal_leq_thresholds_begin());
            ordinalGr_.copyConditions(ruleIndex, bodyPtr->ordinal_gr_indices_begin(),
                                      bodyPtr->ordinal_gr_thresholds_begin());
            nominalEq_.copyConditions(ruleIndex, bodyPtr->nominal_eq_indices_begin(),
                                      bodyPtr->nominal_eq_thresholds_begin());
            nominalNeq_.copyConditions(ruleIndex, bodyPtr->nominal_neq_indices_begin(),
                                       bodyPtr->nominal_neq_thresholds_begin());
            return bodyPtr;
        }

        /**
         * Returns whether the body of a specific rule does not contain any conditions.
         */
        bool hasNoConditions(uint32 ruleIndex) const {
            return numericalLeq_.getNumConditions(ruleIndex) == 0 && numericalGr_.getNumConditions(ruleIndex) == 0
                   && ordinalLeq_.getNumConditions(ruleIndex) == 0 && ordinalGr_.getNumConditions(ruleIndex) == 0
                   && nominalEq_.getNumConditions(ruleIndex) == 0 && nominalNeq_.getNumConditions(ruleIndex) == 0;
        }
};

/**
 * Returns the name of the score type that corresponds to a specific C++ type.
 */
template<typename ScoreType>
static const char* getScoreTypeName() {
    if constexpr (std::is_same_v<ScoreType, uint8>) {
        return SCORE_TYPE_UINT8;
    } else if constexpr (std::is_same_v<ScoreType, float32>) {
        return SCORE_TYPE_FLOAT32;
    } else {
        static_assert(std::is_same_v<ScoreType, float64>, "Unsupported score type");
        return SCORE_TYPE_FLOAT64;
    }
}

/**
 * Collects the heads of all rules in a columnar format, while visiting the rules.
 */
class HeadCollector final {
    private:

        const char* scoreType_;

        std::vector<uint8> headTypes_;

        std::vector<uint32> scoreOffsets_;

        std::vector<uint32> indexOffsets_;

        std::vector<uint32> outputIndices_;

        std::vector<uint8> binaryScores_;

        std::vector<float32> scores32Bit_;

        std::vector<float64> scores64Bit_;

        template<typename ScoreType>
        std::vector<ScoreType>& getScores() {
            if constexpr (std::is_same_v<ScoreType, uint8>) {
                return binaryScores_;
            } else if constexpr (std::is_same_v<ScoreType, float32>) {
                return scores32Bit_;
            } else {
                return scores64Bit_;
            }
        }

        template<typename ScoreType, typename ValueIterator>
        void addScores(uint8 headType, uint32 numScores, ValueIterator values) {
            const char* scoreType = getScoreTypeName<ScoreType>();

            if (!scoreType_) {
                scoreType_ = scoreType;
            } else if (scoreType_ != scoreType) {
                throw std::runtime_error(
                  "Unable to write rule model: All rules must use the same type of scores, but both \""
                  + std::string(scoreType_) + "\" and \"" + std::string(scoreType) + "\" are used");
            }

            std::vector<ScoreType>& scores = this->getScores<ScoreType>();

            for (uint32 i = 0; i < numScores; i++) {
                scores.push_back(values[i]);
            }

            headTypes_.push_back(headType);
            scoreOffsets_.push_back(toOffset(scores.size()));
        }

    public:

        HeadCollector() : scoreType_(nullptr), scoreOffsets_({0}), indexOffsets_({0}) {}

        template<typename ScoreType>
        void addCompleteHead(const CompleteHead<ScoreType>& head) {
            this->addScores<ScoreType>(HEAD_TYPE_COMPLETE, head.getNumElements(), head.values_cbegin());
            indexOffsets_.push_back(toOffset(outputIndices_.size()));
        }

        template<typename ScoreType>
        void addPartialHead(const PartialHead<ScoreType>& head) {
            uint32 numElements = head.getNumElements();
            this->addScores<ScoreType>(HEAD_TYPE_PARTIAL, numElements, head.values_cbegin());
            auto indices = head.indices_cbegin();

            for (uint32 i = 0; i < numElements; i++) {
                outputIndices_.push_back(indices[i]);
            }

            indexOffsets_.push_back(toOffset(outputIndices_.size()));
        }

        void write(Hdf5Group& group) const {
            Hdf5Group heads = group.createGroup(GROUP_HEADS);
            // If the model does not contain any rules, the score type is irrelevant
            const char* scoreType = scoreType_ ? scoreType_ : SCORE_TYPE_FLOAT64;
            heads.setStringAttribute(ATTRIBUTE_SCORE_TYPE, scoreType);
            heads.writeDataset<uint8>(DATASET_HEAD_TYPE, headTypes_.data(), headTypes_.size());
            heads.writeDataset<uint32>(DATASET_SCORE_OFFSETS, scoreOffsets_.data(), scoreOffsets_.size());

            if (scoreType == SCORE_TYPE_UINT8) {
                heads.writeDataset<uint8>(DATASET_SCORES, binaryScores_.data(), binaryScores_.size());
            } else if (scoreType == SCORE_TYPE_FLOAT32) {
                heads.writeDataset<float32>(DATASET_SCORES, scores32Bit_.data(), scores32Bit_.size());
            } else {
                heads.writeDataset<float64>(DATASET_SCORES, scores64Bit_.data(), scores64Bit_.size());
            }

            heads.writeDataset<uint32>(DATASET_INDEX_OFFSETS, indexOffsets_.data(), indexOffsets_.size());
            heads.writeDataset<uint32>(DATASET_OUTPUT_INDICES, outputIndices_.data(), outputIndices_.size());
        }
};

/**
 * Provides access to the heads of all rules that have been read from a group.
 *
 * @tparam ScoreType The type of the scores that are used by the heads
 */
template<typename ScoreType>
class HeadColumns final {
    private:

        std::vector<uint8> headTypes_;

        std::vector<uint32> scoreOffsets_;

        std::vector<ScoreType> scores_;

        std::vector<uint32> indexOffsets_;

        std::vector<uint32> outputIndices_;

    public:

        HeadColumns(const Hdf5Group& heads, std::size_t numRules)
            : headTypes_(heads.readDataset<uint8>(DATASET_HEAD_TYPE)),
              scoreOffsets_(heads.readDataset<uint32>(DATASET_SCORE_OFFSETS)),
              scores_(heads.readDataset<ScoreType>(DATASET_SCORES)),
              indexOffsets_(heads.readDataset<uint32>(DATASET_INDEX_OFFSETS)),
              outputIndices_(heads.readDataset<uint32>(DATASET_OUTPUT_INDICES)) {
            if (headTypes_.size() != numRules) {
                throwInvalidRuleModel("The number of head types does not match the number of rules");
            }

            validateOffsets(scoreOffsets_, numRules, scores_.size(), "the scores");
            validateOffsets(indexOffsets_, numRules, outputIndices_.size(), "the output indices");
        }

        /**
         * Creates and returns the head of a specific rule.
         */
        std::unique_ptr<IHead> createHead(uint32 ruleIndex) const {
            uint32 scoreStart = scoreOffsets_[ruleIndex];
            uint32 numScores = scoreOffsets_[ruleIndex + 1] - scoreStart;
            uint32 indexStart = indexOffsets_[ruleIndex];
            uint32 numIndices = indexOffsets_[ruleIndex + 1] - indexStart;
            uint8 headType = headTypes_[ruleIndex];

            if (headType == HEAD_TYPE_COMPLETE) {
                if (numIndices != 0) {
                    throwInvalidRuleModel("Rule " + std::to_string(ruleIndex)
                                          + " has a complete head, but output indices are specified");
                }

                std::unique_ptr<CompleteHead<ScoreType>> headPtr = std::make_unique<CompleteHead<ScoreType>>(numScores);
                auto values = headPtr->values_begin();

                for (uint32 i = 0; i < numScores; i++) {
                    values[i] = scores_[scoreStart + i];
                }

                return headPtr;
            } else if (headType == HEAD_TYPE_PARTIAL) {
                if (numIndices != numScores) {
                    throwInvalidRuleModel("Rule " + std::to_string(ruleIndex)
                                          + " has a partial head with different numbers of scores and output indices");
                }

                std::unique_ptr<PartialHead<ScoreType>> headPtr = std::make_unique<PartialHead<ScoreType>>(numScores);
                auto values = headPtr->values_begin();
                auto indices = headPtr->indices_begin();

                for (uint32 i = 0; i < numScores; i++) {
                    values[i] = scores_[scoreStart + i];
                    indices[i] = outputIndices_[indexStart + i];
                }

                return headPtr;
            }

            throwInvalidRuleModel("Rule " + std::to_string(ruleIndex) + " has a head of unknown type "
                                  + std::to_string(headType));
        }
};

template<typename ScoreType>
static std::unique_ptr<IRuleList> readRuleList(const Hdf5Group& group, const Hdf5Group& heads, uint32 numRules,
                                               uint32 numUsedRules, bool defaultRuleTakesPrecedence) {
    BodyColumns bodies;
    bodies.read(group, numRules);
    HeadColumns<ScoreType> headColumns(heads, numRules);
    const std::vector<uint8>& bodyTypes = bodies.getBodyTypes();
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(defaultRuleTakesPrecedence);

    for (uint32 i = 0; i < numRules; i++) {
        uint8 bodyType = bodyTypes[i];

        if (bodyType == BODY_TYPE_EMPTY) {
            // The default rule must be the first rule, if it takes precedence, or the last rule otherwise
            uint32 expectedIndex = defaultRuleTakesPrecedence ? 0 : numRules - 1;

            if (i != expectedIndex) {
                throwInvalidRuleModel("Rule " + std::to_string(i)
                                      + " has an empty body, but only the default rule may have an empty body and it "
                                      + "must be the " + (defaultRuleTakesPrecedence ? "first" : "last") + " rule");
            }

            if (!bodies.hasNoConditions(i)) {
                throwInvalidRuleModel("Rule " + std::to_string(i) + " has an empty body, but conditions are specified");
            }

            ruleListPtr->addDefaultRule(headColumns.createHead(i));
        } else if (bodyType == BODY_TYPE_CONJUNCTIVE) {
            ruleListPtr->addRule(bodies.createConjunctiveBody(i), headColumns.createHead(i));
        } else {
            throwInvalidRuleModel("Rule " + std::to_string(i) + " has a body of unknown type "
                                  + std::to_string(bodyType));
        }
    }

    ruleListPtr->setNumUsedRules(numUsedRules);
    return ruleListPtr;
}

static void writeRuleList(Hdf5Group& parent, const IRuleList& ruleList) {
    BodyColumns bodies;
    HeadCollector heads;
    ruleList.visit([&](const EmptyBody& body) {
        bodies.addEmptyBody(body);
    }, [&](const ConjunctiveBody& body) {
        bodies.addConjunctiveBody(body);
    }, [&](const CompleteHead<uint8>& head) {
        heads.addCompleteHead(head);
    }, [&](const CompleteHead<float32>& head) {
        heads.addCompleteHead(head);
    }, [&](const CompleteHead<float64>& head) {
        heads.addCompleteHead(head);
    }, [&](const PartialHead<uint8>& head) {
        heads.addPartialHead(head);
    }, [&](const PartialHead<float32>& head) {
        heads.addPartialHead(head);
    }, [&](const PartialHead<float64>& head) {
        heads.addPartialHead(head);
    });

    // Only the default rule may have an empty body. Otherwise, the rule list could not be restored when reading it
    const std::vector<uint8>& bodyTypes = bodies.getBodyTypes();
    uint32 numRules = toOffset(bodyTypes.size());
    bool defaultRuleTakesPrecedence = ruleList.isDefaultRuleTakingPrecedence();
    uint32 numEmptyBodies = 0;

    for (uint8 bodyType : bodyTypes) {
        if (bodyType == BODY_TYPE_EMPTY) {
            numEmptyBodies++;
        }
    }

    if (ruleList.containsDefaultRule()) {
        uint32 defaultRuleIndex = defaultRuleTakesPrecedence ? 0 : numRules - 1;

        if (numEmptyBodies != 1 || bodyTypes[defaultRuleIndex] != BODY_TYPE_EMPTY) {
            throw std::runtime_error("Unable to write rule model: Only the default rule may have an empty body");
        }
    } else if (numEmptyBodies > 0) {
        throw std::runtime_error("Unable to write rule model: Only the default rule may have an empty body");
    }

    Hdf5Group group = parent.createGroup(GROUP_RULE_MODEL);
    group.setStringAttribute(ATTRIBUTE_TYPE, TYPE_RULE_LIST);
    group.setIntAttribute(ATTRIBUTE_DEFAULT_RULE_TAKES_PRECEDENCE, defaultRuleTakesPrecedence ? 1 : 0);
    // The number of used rules may exceed the number of available rules, in which case all rules are used
    group.setIntAttribute(ATTRIBUTE_NUM_USED_RULES, std::min(ruleList.getNumUsedRules(), numRules));
    group.setIntAttribute(ATTRIBUTE_NUM_RULES, numRules);
    bodies.write(group);
    heads.write(group);
}

void writeRuleModel(Hdf5Group& parent, const IRuleModel& ruleModel) {
    const IRuleList* ruleList = dynamic_cast<const IRuleList*>(&ruleModel);

    if (!ruleList) {
        throw std::runtime_error("Unable to write rule model of unsupported type");
    }

    writeRuleList(parent, *ruleList);
}

std::unique_ptr<IRuleModel> readRuleModel(const Hdf5Group& parent) {
    Hdf5Group group = parent.openGroup(GROUP_RULE_MODEL);
    std::string type = group.getStringAttribute(ATTRIBUTE_TYPE);

    if (type != TYPE_RULE_LIST) {
        throwInvalidRuleModel("Unknown type \"" + type + "\"");
    }

    std::int64_t defaultRuleTakesPrecedence = group.getIntAttribute(ATTRIBUTE_DEFAULT_RULE_TAKES_PRECEDENCE);
    std::int64_t numRules = group.getIntAttribute(ATTRIBUTE_NUM_RULES);
    std::int64_t numUsedRules = group.getIntAttribute(ATTRIBUTE_NUM_USED_RULES);

    if (defaultRuleTakesPrecedence != 0 && defaultRuleTakesPrecedence != 1) {
        throwInvalidRuleModel("The attribute \"" + std::string(ATTRIBUTE_DEFAULT_RULE_TAKES_PRECEDENCE)
                              + "\" must be 0 or 1");
    }

    if (numRules < 0 || numRules > std::numeric_limits<uint32>::max()) {
        throwInvalidRuleModel("The number of rules is invalid");
    }

    if (numUsedRules < 0 || numUsedRules > numRules) {
        throwInvalidRuleModel("The number of used rules is invalid");
    }

    Hdf5Group heads = group.openGroup(GROUP_HEADS);
    std::string scoreType = heads.getStringAttribute(ATTRIBUTE_SCORE_TYPE);
    uint32 numRules32 = static_cast<uint32>(numRules);
    uint32 numUsedRules32 = static_cast<uint32>(numUsedRules);
    bool precedence = defaultRuleTakesPrecedence == 1;

    if (scoreType == SCORE_TYPE_UINT8) {
        return readRuleList<uint8>(group, heads, numRules32, numUsedRules32, precedence);
    } else if (scoreType == SCORE_TYPE_FLOAT32) {
        return readRuleList<float32>(group, heads, numRules32, numUsedRules32, precedence);
    } else if (scoreType == SCORE_TYPE_FLOAT64) {
        return readRuleList<float64>(group, heads, numRules32, numUsedRules32, precedence);
    }

    throwInvalidRuleModel("Unknown score type \"" + scoreType + "\"");
}
