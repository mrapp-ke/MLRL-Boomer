/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <algorithm>
#include <regex>
#include <string>

namespace util {

    /**
     * Converts a given string into lower-case characters.
     *
     * @param string A reference to an `std::string` that should be modified
     * @return       A reference to the modified string
     */
    static inline std::string& convertToLowerCase(std::string& string) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        return string;
    }

    /**
     * Replaces all substrings in a given string matching a specific regular expression with a replacement string.
     *
     * @param string        A reference to an `std::string` that should be modified
     * @param regex         A reference to an `std::string` that specifies the regular expression
     * @param replacement   A reference to an `std::string` to replace matches with
     * @return              The modified string
     */
    static inline std::string replaceRegex(const std::string& string, const std::string& regex,
                                           const std::string& replacement) {
        return std::regex_replace(string, std::regex {regex}, replacement);
    }

    /**
     * Replaces all whitespace in a given string with a replacement string.
     *
     * @param string        A reference to an `std:.string` that should be modified
     * @param replacement   A reference to an `std::string` to replace whitespace with
     * @return              The modified string
     */
    static inline std::string replaceWhitespace(const std::string& string, const std::string& replacement) {
        return replaceRegex(string, " ", replacement);
    }

}
