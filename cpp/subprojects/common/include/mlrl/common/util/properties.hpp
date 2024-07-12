/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <functional>
#include <memory>
#include <utility>

/**
 * A getter function.
 *
 * @tparam T The return type of the getter function
 */
template<typename T>
using GetterFunction = std::function<T&()>;

/**
 * A setter function.
 *
 * @tparam T The argument type of the setter function
 */
template<typename T>
using SetterFunction = std::function<void(std::unique_ptr<T>&&)>;

/**
 * Provides access to a property via a getter and setter function.
 *
 * @tparam T The type of the property
 */
template<typename T>
struct Property {
    public:

        /**
         * The getter function.
         */
        const GetterFunction<T> get;

        /**
         * The setter function.
         */
        const SetterFunction<T> set;

        /**
         * @param getterFunction    The getter function
         * @param setterFunction    The setter function
         */
        Property(GetterFunction<T> getterFunction, SetterFunction<T> setterFunction)
            : get(getterFunction), set(setterFunction) {}
};

/**
 * Creates and returns a `GetterFunction` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `GetterFunction` that has been created
 */
template<typename T>
static inline GetterFunction<T> getterFunction(const std::unique_ptr<T>& uniquePtr) {
    return [&uniquePtr]() -> T& {
        return *uniquePtr;
    };
}

/**
 * Creates and returns a `SetterFunction` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `SetterFunction` that has been created
 */
template<typename T>
static inline SetterFunction<T> setterFunction(std::unique_ptr<T>& uniquePtr) {
    return [&uniquePtr](std::unique_ptr<T>&& ptr) {
        uniquePtr = std::move(ptr);
    };
}

/**
 * Creates and returns a `Property` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `Property` that has been created
 */
template<typename T>
static inline Property<T> property(std::unique_ptr<T>& uniquePtr) {
    return Property<T>(getterFunction(uniquePtr), setterFunction(uniquePtr));
}
