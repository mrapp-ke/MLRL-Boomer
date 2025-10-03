/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <functional>
#include <memory>

namespace util {

    /**
     * An unique pointer that is initialized lazily once it is dereferenced.
     *
     * @tparam T The type of the unqiue pointer
     */
    template<typename T>
    class lazy_unique_ptr final {
        public:

            /**
             * The type of a function that may be used to initialize the unique pointer.
             */
            typedef std::function<std::unique_ptr<T>()> FactoryFunction;

        private:

            FactoryFunction factoryFunction_;

            std::unique_ptr<T> ptr_;

            void initialize() {
                if (!ptr_) {
                    ptr_ = factoryFunction_();
                }
            }

        public:

            lazy_unique_ptr()
                : factoryFunction_([] {
                      return std::make_unique<T>();
                  }) {}

            /**
             * @param factoryFunction The function to be invoked in order to initialize the unique pointer
             */
            explicit lazy_unique_ptr(FactoryFunction factoryFunction) : factoryFunction_(factoryFunction) {}

            /**
             * Returns a pointer to the value of the unique pointer.
             *
             * @return A pointer to the value
             */
            T* get() {
                initialize();
                return ptr_.get();
            }

            /**
             * Returns a pointer to the value of the unique pointer.
             *
             * @return A pointer to the value
             */
            T* operator->() {
                initialize();
                return ptr_.get();
            }

            /**
             * Returns a reference to the value of the unique pointer.
             *
             * @return A reference to the value
             */
            T& operator*() {
                initialize();
                return *ptr_;
            }

            /**
             * Returns whether the unique pointer has already been initialized or not.
             *
             * @return True, if the unique pointer has been initialized, false otherwise
             */
            operator bool() const {
                return ptr_ != nullptr;
            }
    };

}
