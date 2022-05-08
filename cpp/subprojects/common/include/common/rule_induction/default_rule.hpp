/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once


/**
 * Defines an interface for all classes that allow to configure the default rule that is included in a rule-based model.
 */
class IDefaultRuleConfig {

    public:

        virtual ~IDefaultRuleConfig() { };

        /**
         * Returns whether a default rule is included or not.
         *
         * @return True, if a default rule is included, false otherwise
         */
        virtual bool isDefaultRuleUsed() const = 0;

};

/**
 * Allows to configure whether a default rule should be included in a rule-based model or not.
 */
class DefaultRuleConfig final : public IDefaultRuleConfig {

    private:

        bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be included, false otherwise
         */
        DefaultRuleConfig(bool useDefaultRule);

        bool isDefaultRuleUsed() const override;

};
