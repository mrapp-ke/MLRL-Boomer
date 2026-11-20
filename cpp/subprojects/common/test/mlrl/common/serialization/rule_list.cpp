#include "mlrl/common/model/rule_list.hpp"

#include "mlrl/common/model/body_conjunctive.hpp"
#include "mlrl/common/model/body_empty.hpp"
#include "mlrl/common/model/head_complete.hpp"
#include "mlrl/common/model/head_partial.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/serialization/hdf5/hdf5_file.hpp"
#include "mlrl/common/serialization/model_file.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

template<typename Threshold>
using Conditions = std::vector<std::pair<uint32, Threshold>>;

/**
 * Specifies the conditions of a conjunctive body.
 */
struct BodySpec final {
    public:

        Conditions<float32> numericalLeq;
        Conditions<float32> numericalGr;
        Conditions<int32> ordinalLeq;
        Conditions<int32> ordinalGr;
        Conditions<int32> nominalEq;
        Conditions<int32> nominalNeq;
};

template<typename Threshold, typename IndexIterator, typename ThresholdIterator>
static void copyConditions(const Conditions<Threshold>& conditions, IndexIterator indices,
                           ThresholdIterator thresholds) {
    for (std::size_t i = 0; i < conditions.size(); i++) {
        indices[i] = conditions[i].first;
        thresholds[i] = conditions[i].second;
    }
}

static std::unique_ptr<ConjunctiveBody> createBody(const BodySpec& spec) {
    std::unique_ptr<ConjunctiveBody> bodyPtr = std::make_unique<ConjunctiveBody>(
      static_cast<uint32>(spec.numericalLeq.size()), static_cast<uint32>(spec.numericalGr.size()),
      static_cast<uint32>(spec.ordinalLeq.size()), static_cast<uint32>(spec.ordinalGr.size()),
      static_cast<uint32>(spec.nominalEq.size()), static_cast<uint32>(spec.nominalNeq.size()));
    copyConditions(spec.numericalLeq, bodyPtr->numerical_leq_indices_begin(),
                   bodyPtr->numerical_leq_thresholds_begin());
    copyConditions(spec.numericalGr, bodyPtr->numerical_gr_indices_begin(), bodyPtr->numerical_gr_thresholds_begin());
    copyConditions(spec.ordinalLeq, bodyPtr->ordinal_leq_indices_begin(), bodyPtr->ordinal_leq_thresholds_begin());
    copyConditions(spec.ordinalGr, bodyPtr->ordinal_gr_indices_begin(), bodyPtr->ordinal_gr_thresholds_begin());
    copyConditions(spec.nominalEq, bodyPtr->nominal_eq_indices_begin(), bodyPtr->nominal_eq_thresholds_begin());
    copyConditions(spec.nominalNeq, bodyPtr->nominal_neq_indices_begin(), bodyPtr->nominal_neq_thresholds_begin());
    return bodyPtr;
}

template<typename ScoreType>
static std::unique_ptr<IHead> createCompleteHead(const std::vector<ScoreType>& scores) {
    std::unique_ptr<CompleteHead<ScoreType>> headPtr =
      std::make_unique<CompleteHead<ScoreType>>(static_cast<uint32>(scores.size()));
    std::copy(scores.begin(), scores.end(), headPtr->values_begin());
    return headPtr;
}

template<typename ScoreType>
static std::unique_ptr<IHead> createPartialHead(const std::vector<std::pair<uint32, ScoreType>>& scores) {
    std::unique_ptr<PartialHead<ScoreType>> headPtr =
      std::make_unique<PartialHead<ScoreType>>(static_cast<uint32>(scores.size()));
    auto indices = headPtr->indices_begin();
    auto values = headPtr->values_begin();

    for (std::size_t i = 0; i < scores.size(); i++) {
        indices[i] = scores[i].first;
        values[i] = scores[i].second;
    }

    return headPtr;
}

template<typename Iterator>
static void describeConditions(std::ostream& stream, const char* name, uint32 numConditions,
                               ConjunctiveBody::index_const_iterator indices, Iterator thresholds) {
    stream << name << "[";

    for (uint32 i = 0; i < numConditions; i++) {
        stream << indices[i] << ":" << thresholds[i] << ",";
    }

    stream << "]";
}

template<typename ScoreType>
static void describeCompleteHead(std::ostream& stream, const CompleteHead<ScoreType>& head) {
    stream << " complete<" << sizeof(ScoreType) << ">[";
    auto values = head.values_cbegin();

    for (uint32 i = 0; i < head.getNumElements(); i++) {
        stream << +values[i] << ",";
    }

    stream << "]\n";
}

template<typename ScoreType>
static void describePartialHead(std::ostream& stream, const PartialHead<ScoreType>& head) {
    stream << " partial<" << sizeof(ScoreType) << ">[";
    auto values = head.values_cbegin();
    auto indices = head.indices_cbegin();

    for (uint32 i = 0; i < head.getNumElements(); i++) {
        stream << indices[i] << ":" << +values[i] << ",";
    }

    stream << "]\n";
}

/**
 * Returns a textual description of a rule list that contains all of its properties, rules, conditions, and scores in
 * the order they are visited. Floating point values are represented exactly.
 */
static std::string describe(const IRuleList& ruleList) {
    std::ostringstream stream;
    stream << std::hexfloat;
    stream << "precedence=" << ruleList.isDefaultRuleTakingPrecedence()
           << ", default=" << ruleList.containsDefaultRule() << ", rules=" << ruleList.getNumRules()
           << ", used=" << ruleList.getNumUsedRules() << "\n";
    ruleList.visit([&](const EmptyBody&) {
        stream << "empty";
    }, [&](const ConjunctiveBody& body) {
        stream << "conjunctive ";
        describeConditions(stream, "numerical_leq", body.getNumNumericalLeq(), body.numerical_leq_indices_cbegin(),
                           body.numerical_leq_thresholds_cbegin());
        describeConditions(stream, "numerical_gr", body.getNumNumericalGr(), body.numerical_gr_indices_cbegin(),
                           body.numerical_gr_thresholds_cbegin());
        describeConditions(stream, "ordinal_leq", body.getNumOrdinalLeq(), body.ordinal_leq_indices_cbegin(),
                           body.ordinal_leq_thresholds_cbegin());
        describeConditions(stream, "ordinal_gr", body.getNumOrdinalGr(), body.ordinal_gr_indices_cbegin(),
                           body.ordinal_gr_thresholds_cbegin());
        describeConditions(stream, "nominal_eq", body.getNumNominalEq(), body.nominal_eq_indices_cbegin(),
                           body.nominal_eq_thresholds_cbegin());
        describeConditions(stream, "nominal_neq", body.getNumNominalNeq(), body.nominal_neq_indices_cbegin(),
                           body.nominal_neq_thresholds_cbegin());
    }, [&](const CompleteHead<uint8>& head) {
        describeCompleteHead(stream, head);
    }, [&](const CompleteHead<float32>& head) {
        describeCompleteHead(stream, head);
    }, [&](const CompleteHead<float64>& head) {
        describeCompleteHead(stream, head);
    }, [&](const PartialHead<uint8>& head) {
        describePartialHead(stream, head);
    }, [&](const PartialHead<float32>& head) {
        describePartialHead(stream, head);
    }, [&](const PartialHead<float64>& head) {
        describePartialHead(stream, head);
    });
    return stream.str();
}

static std::vector<uint8> write(const IRuleModel& ruleModel) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeRuleModel(ruleModel);
    return writer->finish();
}

static std::unique_ptr<IRuleList> read(const std::vector<uint8>& data) {
    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_TRUE(reader->hasRuleModel());
    std::unique_ptr<IRuleModel> ruleModelPtr = reader->readRuleModel();
    IRuleList* ruleList = dynamic_cast<IRuleList*>(ruleModelPtr.get());

    if (!ruleList) {
        ADD_FAILURE() << "Expected an object of type IRuleList";
        return nullptr;
    }

    ruleModelPtr.release();
    return std::unique_ptr<IRuleList>(ruleList);
}

/**
 * Writes a rule list, reads it, and checks whether the original and the loaded rule list are identical.
 */
static void expectRoundTrip(const IRuleList& ruleList) {
    std::unique_ptr<IRuleList> loadedRuleListPtr = read(write(ruleList));
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(describe(*loadedRuleListPtr), describe(ruleList));
}

static BodySpec createBodyWithAllConditionTypes() {
    BodySpec spec;
    spec.numericalLeq = {
      {0, -1.5f},
      {7,  0.1f}
    };
    spec.numericalGr = {
      {2, 3.25f}
    };
    spec.ordinalLeq = {
      {4,         -3},
      {5,         10},
      {6, 2147483647}
    };
    spec.ordinalGr = {
      {8, 1}
    };
    spec.nominalEq = {
      {9, 0}
    };
    spec.nominalNeq = {
      {10, -2147483647 - 1},
      {11,              42}
    };
    return spec;
}

TEST(RuleListSerializationTest, emptyRuleList) {
    expectRoundTrip(*createRuleList(true));
    expectRoundTrip(*createRuleList(false));
}

TEST(RuleListSerializationTest, ruleListWithoutDefaultRule) {
    for (bool defaultRuleTakesPrecedence : {true, false}) {
        std::unique_ptr<IRuleList> ruleListPtr = createRuleList(defaultRuleTakesPrecedence);
        BodySpec spec;
        spec.numericalLeq = {
          {1, 0.5f}
        };
        ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({0.1, -0.2}));
        spec.numericalLeq = {};
        spec.nominalEq = {
          {3, 2}
        };
        ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({0.3, 0.4}));
        expectRoundTrip(*ruleListPtr);
    }
}

TEST(RuleListSerializationTest, defaultRuleTakesPrecedence) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(true);
    ruleListPtr->addDefaultRule(createCompleteHead<float64>({0.5, -0.5, 0.25}));
    ruleListPtr->addRule(createBody(createBodyWithAllConditionTypes()), createCompleteHead<float64>({1.0, 2.0, 3.0}));
    BodySpec spec;
    spec.ordinalGr = {
      {1, 5}
    };
    ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({-1.0, 0.0, 1.0}));

    std::unique_ptr<IRuleList> loadedRuleListPtr = read(write(*ruleListPtr));
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(describe(*loadedRuleListPtr), describe(*ruleListPtr));
    EXPECT_TRUE(loadedRuleListPtr->containsDefaultRule());
    EXPECT_TRUE(loadedRuleListPtr->isDefaultRuleTakingPrecedence());
    EXPECT_EQ(loadedRuleListPtr->getNumRules(), (uint32) 3);
}

TEST(RuleListSerializationTest, defaultRuleDoesNotTakePrecedence) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    ruleListPtr->addDefaultRule(createCompleteHead<float64>({0.5, -0.5}));
    ruleListPtr->addRule(createBody(createBodyWithAllConditionTypes()), createCompleteHead<float64>({1.0, 2.0}));

    std::unique_ptr<IRuleList> loadedRuleListPtr = read(write(*ruleListPtr));
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(describe(*loadedRuleListPtr), describe(*ruleListPtr));
    EXPECT_TRUE(loadedRuleListPtr->containsDefaultRule());
    EXPECT_FALSE(loadedRuleListPtr->isDefaultRuleTakingPrecedence());
}

TEST(RuleListSerializationTest, ruleListWithOnlyDefaultRule) {
    for (bool defaultRuleTakesPrecedence : {true, false}) {
        std::unique_ptr<IRuleList> ruleListPtr = createRuleList(defaultRuleTakesPrecedence);
        ruleListPtr->addDefaultRule(createCompleteHead<float32>({0.5f}));
        expectRoundTrip(*ruleListPtr);
    }
}

TEST(RuleListSerializationTest, bodiesWithSomeOrNoConditions) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    ruleListPtr->addRule(createBody(createBodyWithAllConditionTypes()), createCompleteHead<float64>({1.0}));
    ruleListPtr->addRule(createBody(BodySpec()), createCompleteHead<float64>({2.0}));
    BodySpec spec;
    spec.numericalGr = {
      {0, 1.0f},
      {1, 2.0f},
      {2, 3.0f}
    };
    ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({3.0}));
    spec = BodySpec();
    spec.nominalNeq = {
      {4, 7}
    };
    ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({4.0}));
    ruleListPtr->addDefaultRule(createCompleteHead<float64>({0.0}));
    expectRoundTrip(*ruleListPtr);
}

template<typename ScoreType>
static void testHeadsWithScoreType(const std::vector<ScoreType>& values) {
    for (bool defaultRuleTakesPrecedence : {true, false}) {
        std::unique_ptr<IRuleList> ruleListPtr = createRuleList(defaultRuleTakesPrecedence);
        ruleListPtr->addDefaultRule(createCompleteHead<ScoreType>({values[0], values[1], values[2]}));
        BodySpec spec;
        spec.numericalLeq = {
          {0, 0.5f}
        };
        ruleListPtr->addRule(createBody(spec), createPartialHead<ScoreType>({
                                                 {2, values[1]}
        }));
        ruleListPtr->addRule(createBody(spec), createCompleteHead<ScoreType>({values[2], values[0], values[1]}));
        ruleListPtr->addRule(createBody(spec), createPartialHead<ScoreType>({
                                                 {0, values[0]},
                                                 {1, values[2]}
        }));
        ruleListPtr->addRule(createBody(spec), createPartialHead<ScoreType>({}));
        expectRoundTrip(*ruleListPtr);
    }
}

TEST(RuleListSerializationTest, headsWithBinaryScores) {
    testHeadsWithScoreType<uint8>({0, 1, 255});
}

TEST(RuleListSerializationTest, headsWith32BitScores) {
    testHeadsWithScoreType<float32>({-0.1f, 0.0f, 1e-30f});
}

TEST(RuleListSerializationTest, headsWith64BitScores) {
    testHeadsWithScoreType<float64>({-0.1, 0.0, 1e-300});
}

TEST(RuleListSerializationTest, numUsedRules) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    BodySpec spec;
    spec.numericalLeq = {
      {0, 0.5f}
    };

    for (uint32 i = 0; i < 5; i++) {
        ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({static_cast<float64>(i)}));
    }

    ruleListPtr->addDefaultRule(createCompleteHead<float64>({0.0}));
    ruleListPtr->setNumUsedRules(3);
    std::unique_ptr<IRuleList> loadedRuleListPtr = read(write(*ruleListPtr));
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(describe(*loadedRuleListPtr), describe(*ruleListPtr));
    EXPECT_EQ(loadedRuleListPtr->getNumRules(), (uint32) 6);
    EXPECT_EQ(loadedRuleListPtr->getNumUsedRules(), (uint32) 3);

    // A number of used rules that exceeds the number of rules is limited to the number of rules
    ruleListPtr->setNumUsedRules(100);
    loadedRuleListPtr = read(write(*ruleListPtr));
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(loadedRuleListPtr->getNumUsedRules(), (uint32) 6);
}

TEST(RuleListSerializationTest, ruleModelCanBeStoredWithOtherModels) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(true);
    ruleListPtr->addDefaultRule(createCompleteHead<float32>({0.5f, 0.25f}));
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->setStringAttribute("learner", "class", "mlrl.boosting.BoomerClassifier");
    writer->writeRuleModel(*ruleListPtr);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_EQ(reader->getStringAttribute("learner", "class"), "mlrl.boosting.BoomerClassifier");
    EXPECT_FALSE(reader->hasOutputSpaceInfo());
    std::unique_ptr<IRuleModel> loadedPtr = reader->readRuleModel();
    IRuleList* loadedRuleList = dynamic_cast<IRuleList*>(loadedPtr.get());
    ASSERT_NE(loadedRuleList, nullptr);
    EXPECT_EQ(describe(*loadedRuleList), describe(*ruleListPtr));
}

TEST(RuleListSerializationTest, missingRuleModelIsReported) {
    std::vector<uint8> data = createModelFileWriter()->finish();
    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_FALSE(reader->hasRuleModel());
    EXPECT_THROW(reader->readRuleModel(), std::runtime_error);
}

TEST(RuleListSerializationTest, writingMixedScoreTypesThrows) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    BodySpec spec;
    spec.numericalLeq = {
      {0, 0.5f}
    };
    ruleListPtr->addRule(createBody(spec), createCompleteHead<float64>({1.0}));
    ruleListPtr->addRule(createBody(spec), createPartialHead<float32>({
                                             {0, 1.0f}
    }));
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    EXPECT_THROW(writer->writeRuleModel(*ruleListPtr), std::runtime_error);

    // A failed attempt must not leave an incomplete group behind...
    std::vector<uint8> data = writer->finish();
    EXPECT_FALSE(createModelFileReader(data.data(), data.size())->hasRuleModel());
}

TEST(RuleListSerializationTest, writingRuleWithEmptyBodyThrows) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    ruleListPtr->addRule(std::make_unique<EmptyBody>(), createCompleteHead<float64>({1.0}));
    EXPECT_THROW(write(*ruleListPtr), std::runtime_error);
}

TEST(RuleListSerializationTest, writingRuleModelTwiceThrows) {
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeRuleModel(*ruleListPtr);
    EXPECT_THROW(writer->writeRuleModel(*ruleListPtr), std::runtime_error);
}

/**
 * A rule-based model that is not supported by the model file writer.
 */
class UnsupportedRuleModel final : public IRuleModel {
    public:

        uint32 getNumRules() const override {
            return 0;
        }

        uint32 getNumUsedRules() const override {
            return 0;
        }

        void setNumUsedRules(uint32 numUsedRules) override {}

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousView<const float32>& featureMatrix,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override {
            return nullptr;
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrView<const float32>& featureMatrix,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override {
            return nullptr;
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }
};

TEST(RuleListSerializationTest, writingUnsupportedRuleModelThrows) {
    UnsupportedRuleModel ruleModel;
    EXPECT_THROW(write(ruleModel), std::runtime_error);
}

/**
 * The raw content of a group that stores a rule list with two rules: A rule with a single numerical condition and a
 * partial head, followed by a default rule with a complete head.
 */
struct RawRuleModel final {
    public:

        std::string type = "RuleList";
        std::int64_t defaultRuleTakesPrecedence = 0;
        std::int64_t numRules = 2;
        std::int64_t numUsedRules = 2;
        std::vector<uint8> bodyTypes = {1, 0};
        std::vector<uint32> conditionOffsets = {0, 1, 1};
        std::vector<uint32> featureIndices = {3};
        std::vector<float32> thresholds = {0.5f};
        std::string scoreType = "float64";
        std::vector<uint8> headTypes = {1, 0};
        std::vector<uint32> scoreOffsets = {0, 1, 3};
        std::vector<float64> scores = {0.1, 0.2, 0.3};
        std::vector<uint32> indexOffsets = {0, 1, 1};
        std::vector<uint32> outputIndices = {1};
};

static std::unique_ptr<IModelFileReader> createRawFile(const RawRuleModel& raw) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    root.setIntAttribute("format_version", MODEL_FILE_FORMAT_VERSION);
    root.setStringAttribute("library_version", "0.0.0");
    Hdf5Group group = root.createGroup("rule_model");
    group.setStringAttribute("type", raw.type);
    group.setIntAttribute("default_rule_takes_precedence", raw.defaultRuleTakesPrecedence);
    group.setIntAttribute("num_rules", raw.numRules);
    group.setIntAttribute("num_used_rules", raw.numUsedRules);
    group.writeDataset<uint8>("body_type", raw.bodyTypes.data(), raw.bodyTypes.size());
    group.writeDataset<uint32>("bodies/numerical_leq/offsets", raw.conditionOffsets.data(),
                               raw.conditionOffsets.size());
    group.writeDataset<uint32>("bodies/numerical_leq/feature_indices", raw.featureIndices.data(),
                               raw.featureIndices.size());
    group.writeDataset<float32>("bodies/numerical_leq/thresholds", raw.thresholds.data(), raw.thresholds.size());
    std::vector<uint32> emptyOffsets(raw.bodyTypes.size() + 1, 0);

    for (const char* name : {"numerical_gr", "ordinal_leq", "ordinal_gr", "nominal_eq", "nominal_neq"}) {
        std::string prefix = "bodies/" + std::string(name) + "/";
        group.writeDataset<uint32>(prefix + "offsets", emptyOffsets.data(), emptyOffsets.size());
        group.writeDataset<uint32>(prefix + "feature_indices", nullptr, 0);

        if (std::string(name) == "numerical_gr") {
            group.writeDataset<float32>(prefix + "thresholds", nullptr, 0);
        } else {
            group.writeDataset<int32>(prefix + "thresholds", nullptr, 0);
        }
    }

    Hdf5Group heads = group.createGroup("heads");
    heads.setStringAttribute("score_type", raw.scoreType);
    heads.writeDataset<uint8>("head_type", raw.headTypes.data(), raw.headTypes.size());
    heads.writeDataset<uint32>("score_offsets", raw.scoreOffsets.data(), raw.scoreOffsets.size());
    heads.writeDataset<float64>("scores", raw.scores.data(), raw.scores.size());
    heads.writeDataset<uint32>("index_offsets", raw.indexOffsets.data(), raw.indexOffsets.size());
    heads.writeDataset<uint32>("output_indices", raw.outputIndices.data(), raw.outputIndices.size());
    std::vector<uint8> data = file.getImage();
    return createModelFileReader(data.data(), data.size());
}

TEST(RuleListSerializationTest, readingRawRuleModel) {
    std::unique_ptr<IRuleModel> ruleModelPtr = createRawFile(RawRuleModel())->readRuleModel();
    IRuleList* ruleList = dynamic_cast<IRuleList*>(ruleModelPtr.get());
    ASSERT_NE(ruleList, nullptr);

    // The raw model corresponds to the following rule list...
    std::unique_ptr<IRuleList> expectedRuleListPtr = createRuleList(false);
    BodySpec spec;
    spec.numericalLeq = {
      {3, 0.5f}
    };
    expectedRuleListPtr->addRule(createBody(spec), createPartialHead<float64>({
                                                     {1, 0.1}
    }));
    expectedRuleListPtr->addDefaultRule(createCompleteHead<float64>({0.2, 0.3}));
    EXPECT_EQ(describe(*ruleList), describe(*expectedRuleListPtr));
}

TEST(RuleListSerializationTest, readingCorruptedRuleModelThrows) {
    std::vector<std::pair<std::string, RawRuleModel>> cases;
    RawRuleModel raw;

    raw = RawRuleModel();
    raw.type = "Unknown";
    cases.emplace_back("unknown type", raw);

    raw = RawRuleModel();
    raw.scoreType = "int8";
    cases.emplace_back("unknown score type", raw);

    raw = RawRuleModel();
    raw.scoreType = "float32";
    cases.emplace_back("score type does not match dataset", raw);

    raw = RawRuleModel();
    raw.defaultRuleTakesPrecedence = 2;
    cases.emplace_back("invalid precedence", raw);

    raw = RawRuleModel();
    raw.numRules = 3;
    cases.emplace_back("number of rules does not match datasets", raw);

    raw = RawRuleModel();
    raw.numRules = -1;
    cases.emplace_back("negative number of rules", raw);

    raw = RawRuleModel();
    raw.numUsedRules = 3;
    cases.emplace_back("number of used rules exceeds number of rules", raw);

    raw = RawRuleModel();
    raw.bodyTypes = {2, 0};
    cases.emplace_back("unknown body type", raw);

    raw = RawRuleModel();
    raw.bodyTypes = {0, 0};
    cases.emplace_back("two default rules", raw);

    raw = RawRuleModel();
    raw.defaultRuleTakesPrecedence = 1;
    cases.emplace_back("default rule at wrong position", raw);

    raw = RawRuleModel();
    raw.bodyTypes = {1, 0};
    raw.conditionOffsets = {0, 0, 1};
    cases.emplace_back("default rule with conditions", raw);

    raw = RawRuleModel();
    raw.conditionOffsets = {0, 2, 1};
    cases.emplace_back("condition offsets not increasing", raw);

    raw = RawRuleModel();
    raw.conditionOffsets = {0, 1};
    cases.emplace_back("wrong number of condition offsets", raw);

    raw = RawRuleModel();
    raw.thresholds = {0.5f, 0.6f};
    cases.emplace_back("different number of feature indices and thresholds", raw);

    raw = RawRuleModel();
    raw.headTypes = {1, 2};
    cases.emplace_back("unknown head type", raw);

    raw = RawRuleModel();
    raw.headTypes = {0, 0};
    cases.emplace_back("complete head with output indices", raw);

    raw = RawRuleModel();
    raw.scoreOffsets = {0, 2, 3};
    cases.emplace_back("partial head with different number of scores and indices", raw);

    raw = RawRuleModel();
    raw.scoreOffsets = {0, 4, 3};
    cases.emplace_back("score offsets not increasing", raw);

    raw = RawRuleModel();
    raw.scoreOffsets = {0, 1, 2};
    cases.emplace_back("last score offset does not match number of scores", raw);

    raw = RawRuleModel();
    raw.indexOffsets = {0, 2, 1};
    cases.emplace_back("index offsets not increasing", raw);

    for (const auto& testCase : cases) {
        EXPECT_THROW(createRawFile(testCase.second)->readRuleModel(), std::runtime_error) << testCase.first;
    }
}

TEST(RuleListSerializationTest, largeRuleList) {
    const uint32 numRules = 10000;
    const uint32 numOutputs = 20;
    std::unique_ptr<IRuleList> ruleListPtr = createRuleList(false);

    for (uint32 i = 0; i < numRules; i++) {
        BodySpec spec;

        for (uint32 j = 0; j < 5; j++) {
            spec.numericalLeq.emplace_back(i + j, static_cast<float32>(i) * 0.5f + static_cast<float32>(j));
            spec.nominalNeq.emplace_back(j, static_cast<int32>(i));
        }

        std::vector<std::pair<uint32, float64>> scores;

        for (uint32 j = 0; j < 5; j++) {
            scores.emplace_back((i + j) % numOutputs, static_cast<float64>(i) / 3.0);
        }

        ruleListPtr->addRule(createBody(spec), createPartialHead<float64>(scores));
    }

    ruleListPtr->addDefaultRule(createCompleteHead<float64>(std::vector<float64>(numOutputs, 0.5)));

    auto start = std::chrono::steady_clock::now();
    std::vector<uint8> data = write(*ruleListPtr);
    auto afterWriting = std::chrono::steady_clock::now();
    std::unique_ptr<IRuleList> loadedRuleListPtr = read(data);
    auto afterReading = std::chrono::steady_clock::now();
    ASSERT_NE(loadedRuleListPtr, nullptr);
    EXPECT_EQ(describe(*loadedRuleListPtr), describe(*ruleListPtr));

    auto writeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(afterWriting - start).count();
    auto readMillis = std::chrono::duration_cast<std::chrono::milliseconds>(afterReading - afterWriting).count();
    std::cout << "Writing " << numRules << " rules took " << writeMillis << " ms, reading them took " << readMillis
              << " ms, file size is " << data.size() << " bytes" << std::endl;
    // Generous upper bounds that only detect pathological performance regressions
    EXPECT_LT(writeMillis, 10000);
    EXPECT_LT(readMillis, 10000);
}
