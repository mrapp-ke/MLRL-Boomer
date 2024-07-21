/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <functional>
#include <memory>
#include <utility>

/**
 * Provides access to a property via a getter function.
 *
 * @tparam T The type of the property
 */
template<typename T>
struct ReadableProperty {
    public:

        /**
         * A getter function.
         */
        typedef std::function<T&()> GetterFunction;

        /**
         * The getter function.
         */
        const GetterFunction get;

        /**
         * @param getterFunction The getter function
         */
        explicit ReadableProperty(GetterFunction getterFunction) : get(getterFunction) {}
};

/**
 * Provides access to a property via a setter function.
 *
 * @tparam T The type of the property
 */
template<typename T>
struct WritableProperty {
    public:

        /**
         * A setter function.
         */
        typedef std::function<void(std::unique_ptr<T>&&)> SetterFunction;

        /**
         * The setter function.
         */
        const SetterFunction set;

        /**
         * @param setterFunction The setter function
         */
        explicit WritableProperty(SetterFunction setterFunction) : set(setterFunction) {}
};

/**
 * Provides access to a property via a getter and setter function.
 *
 * @tparam T The type of the property
 */
template<typename T>
struct Property : public ReadableProperty<T>,
                  public WritableProperty<T> {
    public:

        /**
         * @param getterFunction    The getter function
         * @param setterFunction    The setter function
         */
        Property(typename ReadableProperty<T>::GetterFunction getterFunction,
                 typename WritableProperty<T>::SetterFunction setterFunction)
            : ReadableProperty<T>(getterFunction), WritableProperty<T>(setterFunction) {}
};

/**
 * Creates and returns a `GetterFunction` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `GetterFunction` that has been created
 */
template<typename T>
static inline constexpr typename ReadableProperty<T>::GetterFunction getterFunction(
  const std::unique_ptr<T>& uniquePtr) {
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
static inline constexpr typename WritableProperty<T>::SetterFunction setterFunction(std::unique_ptr<T>& uniquePtr) {
    return [&uniquePtr](std::unique_ptr<T>&& ptr) {
        uniquePtr = std::move(ptr);
    };
}

/**
 * Creates and returns a `ReadableProperty` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `ReadableProperty` that has been created
 */
template<typename T>
static inline constexpr ReadableProperty<T> readableProperty(const std::unique_ptr<T>& uniquePtr) {
    return ReadableProperty<T>(getterFunction(uniquePtr));
}

/**
 * Creates and returns a `Property` that is backed by an unique pointer.
 *
 * @tparam T        The type of the unique pointer
 * @param uniquePtr A reference to the unique pointer
 * @return          The `Property` that has been created
 */
template<typename T>
static inline constexpr Property<T> property(std::unique_ptr<T>& uniquePtr) {
    return Property<T>(getterFunction(uniquePtr), setterFunction(uniquePtr));
}
