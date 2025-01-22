/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <iterator>

namespace util {

    /**
     * The type of the values that are accessed by an iterator.
     *
     * @tparam Iterator The type of the iterator
     */
    template<typename Iterator>
    using iterator_value = typename std::iterator_traits<Iterator>::value_type;

}
