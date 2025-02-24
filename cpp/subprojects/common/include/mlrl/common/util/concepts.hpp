/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <concepts>

namespace util {

    /**
     * A concept that is satisfied if and only if `Base` is a template class type that is either `Derived` or a public
     * an unambiguous base of `Derived`.
     *
     * @tparam Derived              The type of the object to be checked
     * @tparam Base                 The type of the base class
     * @tparam TemplateArguments    The template arguments of `Base`
     * @param obj                   The object to be checked
     */
    template<class Derived, template<typename> class Base, typename... TemplateArguments>
    concept derived_from_template_class = requires(Derived obj) {
        []<typename... T>(Base<TemplateArguments..., T...>&) {
        }(obj);
    };

}
