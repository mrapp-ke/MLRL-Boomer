/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
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
 * @tparam T    The type of the property
 * @tparam Ptr  The type of the pointer that is backing the property
 */
template<typename T, typename Ptr = std::unique_ptr<T>>
struct WritableProperty {
    public:

        /**
         * A setter function.
         */
        typedef std::function<void(Ptr&&)> SetterFunction;

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
 * @tparam T    The type of the property
 * @tparam Ptr  The type of the pointer that is backing the property
 */
template<typename T, typename Ptr = std::unique_ptr<T>>
struct Property : public ReadableProperty<T>,
                  public WritableProperty<T, Ptr> {
    public:

        /**
         * @param getterFunction    The getter function
         * @param setterFunction    The setter function
         */
        Property(typename ReadableProperty<T>::GetterFunction getterFunction,
                 typename WritableProperty<T, Ptr>::SetterFunction setterFunction)
            : ReadableProperty<T>(getterFunction), WritableProperty<T, Ptr>(setterFunction) {}
};

/**
 * A `Property` that is backed by a shared pointer.
 */
template<typename T>
using SharedProperty = Property<T, std::shared_ptr<T>>;

/**
 * Creates and returns a `GetterFunction` that is backed by an unique pointer.
 *
 * @tparam T        The return type of the `GetterFunction`
 * @param uniquePtr A reference to the unique pointer
 * @return          The `GetterFunction` that has been created
 */
template<typename T>
static inline constexpr typename ReadableProperty<T>::GetterFunction getterFunction(
  const std::unique_ptr<T>& uniquePtr) {
    return [&uniquePtr]() -> T& {
        if (uniquePtr) {
            return *uniquePtr;
        }

        throw std::runtime_error(
          "Failed to invoke GetterFunction backed by a unique pointer, because the pointer is null");
    };
}

/**
 * Creates and returns a `GetterFunction` that is backed by a shared pointer.
 *
 * @tparam T        The return type of the `GetterFunction`
 * @param sharedPtr A reference to the shared pointer
 * @return          The `GetterFunction` that has been created
 */
template<typename T>
static inline constexpr typename ReadableProperty<T>::GetterFunction getterFunction(
  const std::shared_ptr<T>& sharedPtr) {
    return [&sharedPtr]() -> T& {
        if (sharedPtr) {
            return *sharedPtr;
        }

        throw std::runtime_error(
          "Failed to invoke GetterFunction backed by a shared pointer, because the pointer is null");
    };
}

/**
 * Creates and returns a `GetterFunction` that is backed by two shared pointers.
 *
 * @tparam T            The return type of the `GetterFunction`
 * @tparam S1           The type of the first shared pointer. Must be a supertype of `T`
 * @tparam S2           The type of the second shared pointer. Must be a supertype of `T`
 * @param sharedPtr1    A reference to the first shared pointer
 * @param sharedPtr2    A reference to the second shared pointer
 * @return              The `GetterFunction` that has been created
 */
template<typename T, typename S1, typename S2>
static inline constexpr typename ReadableProperty<T>::GetterFunction getterFunction(
  const std::shared_ptr<S1>& sharedPtr1, const std::shared_ptr<S2>& sharedPtr2) {
    return [&sharedPtr1, &sharedPtr2]() -> T& {
        T* ptr = sharedPtr1.get();

        if (!ptr) {
            ptr = sharedPtr2.get();
        }

        if (ptr) {
            return static_cast<T&>(*ptr);
        }

        throw std::runtime_error(
          "Failed to invoke GetterFunction backed by two shared pointers, because both pointers are null");
    };
}

/**
 * Creates and returns a `SetterFunction` that is backed by an unique pointer.
 *
 * @tparam T        The argument type of the `SetterFunction`
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
 * Creates and returns a `SetterFunction` that is backed by a shared pointer.
 *
 * @tparam T        The argument type of the `SetterFunction`
 * @param sharedPtr A reference to the shared pointer
 * @return          The `SetterFunction` that has been created
 */
template<typename T>
static inline constexpr typename WritableProperty<T, std::shared_ptr<T>>::SetterFunction sharedSetterFunction(
  std::shared_ptr<T>& sharedPtr) {
    return [&sharedPtr](std::shared_ptr<T>&& ptr) {
        sharedPtr = std::move(ptr);
    };
}

/**
 * Creates and returns a `ReadableProperty` that is backed by an unique pointer.
 *
 * @tparam T        The type of the `ReadableProperty`
 * @param uniquePtr A reference to the unique pointer
 * @return          The `ReadableProperty` that has been created
 */
template<typename T>
static inline constexpr ReadableProperty<T> readableProperty(const std::unique_ptr<T>& uniquePtr) {
    return ReadableProperty<T>(getterFunction(uniquePtr));
}

/**
 * Creates and returns a `ReadableProperty` that is backed by a shared pointer.
 *
 * @tparam T        The type of the `ReadableProperty`
 * @tparam S        The type of the shared pointer. Must be a supertype of `T`
 * @param sharedPtr A reference to the shared pointer
 * @return          The `ReadableProperty` that has been created
 */
template<typename T, typename S = T>
static inline constexpr ReadableProperty<T> readableProperty(const std::shared_ptr<S>& sharedPtr) {
    return ReadableProperty<T>(getterFunction(sharedPtr));
}

/**
 * Creates and returns a `ReadableProperty` that is backed by two shared pointers.
 *
 * @tparam T            The type of the `ReadableProperty`
 * @tparam S1           The type of the first shared pointer. Must be a supertype of `T`
 * @tparam S2           The type of the second shared pointer. Must be a supertype of `T`
 * @param sharedPtr1    A reference to the first shared pointer
 * @param sharedPtr2    A reference to the second shared pointer
 * @return              The `ReadableProperty` that has been created
 */
template<typename T, typename S1 = T, typename S2 = T>
static inline constexpr ReadableProperty<T> readableProperty(const std::shared_ptr<S1>& sharedPtr1,
                                                             const std::shared_ptr<S2>& sharedPtr2) {
    return ReadableProperty<T>(getterFunction<T, S1, S2>(sharedPtr1, sharedPtr2));
}

/**
 * Creates and returns a `Property` that is backed by an unique pointer.
 *
 * @tparam T        The type of the `Property`
 * @param uniquePtr A reference to the unique pointer
 * @return          The `Property` that has been created
 */
template<typename T>
static inline constexpr Property<T> property(std::unique_ptr<T>& uniquePtr) {
    return Property<T>(getterFunction(uniquePtr), setterFunction(uniquePtr));
}

/**
 * Creates and returns a `Property` that is backed by a shared pointer.
 *
 * @tparam T        The type of the `Property`
 * @param sharedPtr A reference to the shared pointer
 * @return          The `Property` that has been created
 */
template<typename T>
static inline constexpr SharedProperty<T> sharedProperty(std::shared_ptr<T>& sharedPtr) {
    return SharedProperty<T>(getterFunction(sharedPtr), sharedSetterFunction(sharedPtr));
}
