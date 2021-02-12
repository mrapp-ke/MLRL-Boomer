/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <iterator>


/**
 * An iterator that provides random access to the indices in a continuous range.
 */
class IndexIterator final {

    private:

        uint32 index_;

    public:

        IndexIterator();

        IndexIterator(uint32 index);

        typedef int difference_type;

        typedef uint32 value_type;

        typedef uint32* pointer;

        typedef uint32 reference;

        typedef std::random_access_iterator_tag iterator_category;

        reference operator[](uint32 index) const;

        reference operator*() const;

        IndexIterator& operator++();

        IndexIterator& operator++(int n);

        IndexIterator& operator--();

        IndexIterator& operator--(int n);

        bool operator!=(const IndexIterator& rhs) const;

        difference_type operator-(const IndexIterator& rhs) const;

};
