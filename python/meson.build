project('mlrl-python', meson_version : '>=1.1', default_options : ['cpp_std=c++17', 'buildtype=release', 'werror=true'])

subprojects = get_option('subprojects')

foreach subproject_name : ['common', 'boosting', 'seco']
    if subprojects.length() == 0 or subprojects.contains(subproject_name)
        subproject(subproject_name)
    endif
endforeach
