project('boosting', 'cpp')

# Source files
source_files = [
    'src/mlrl/boosting/binning/label_binning_auto.cpp',
    'src/mlrl/boosting/binning/label_binning_equal_width.cpp',
    'src/mlrl/boosting/binning/label_binning_no.cpp',
    'src/mlrl/boosting/data/matrix_c_contiguous_numeric.cpp',
    'src/mlrl/boosting/data/matrix_sparse_set_numeric.cpp',
    'src/mlrl/boosting/data/vector_statistic_decomposable_dense.cpp',
    'src/mlrl/boosting/data/vector_statistic_decomposable_sparse.cpp',
    'src/mlrl/boosting/data/vector_statistic_non_decomposable_dense.cpp',
    'src/mlrl/boosting/data/view_statistic_non_decomposable_dense.cpp',
    'src/mlrl/boosting/input/feature_binning_auto.cpp',
    'src/mlrl/boosting/losses/loss_decomposable_logistic.cpp',
    'src/mlrl/boosting/losses/loss_decomposable_squared_error.cpp',
    'src/mlrl/boosting/losses/loss_decomposable_squared_hinge.cpp',
    'src/mlrl/boosting/losses/loss_non_decomposable_logistic.cpp',
    'src/mlrl/boosting/losses/loss_non_decomposable_squared_error.cpp',
    'src/mlrl/boosting/losses/loss_non_decomposable_squared_hinge.cpp',
    'src/mlrl/boosting/model/rule_list_builder.cpp',
    'src/mlrl/boosting/multi_threading/parallel_rule_refinement_auto.cpp',
    'src/mlrl/boosting/multi_threading/parallel_statistic_update_auto.cpp',
    'src/mlrl/boosting/post_processing/shrinkage_constant.cpp',
    'src/mlrl/boosting/prediction/discretization_function_probability.cpp',
    'src/mlrl/boosting/prediction/discretization_function_score.cpp',
    'src/mlrl/boosting/prediction/predictor_binary_auto.cpp',
    'src/mlrl/boosting/prediction/predictor_binary_example_wise.cpp',
    'src/mlrl/boosting/prediction/predictor_binary_gfm.cpp',
    'src/mlrl/boosting/prediction/predictor_binary_output_wise.cpp',
    'src/mlrl/boosting/prediction/predictor_probability_auto.cpp',
    'src/mlrl/boosting/prediction/predictor_probability_marginalized.cpp',
    'src/mlrl/boosting/prediction/predictor_probability_output_wise.cpp',
    'src/mlrl/boosting/prediction/predictor_score_output_wise.cpp',
    'src/mlrl/boosting/prediction/probability_calibration_isotonic.cpp',
    'src/mlrl/boosting/prediction/probability_function_chain_rule.cpp',
    'src/mlrl/boosting/prediction/probability_function_logistic.cpp',
    'src/mlrl/boosting/prediction/transformation_binary_example_wise.cpp',
    'src/mlrl/boosting/prediction/transformation_binary_gfm.cpp',
    'src/mlrl/boosting/prediction/transformation_binary_output_wise.cpp',
    'src/mlrl/boosting/prediction/transformation_probability_marginalized.cpp',
    'src/mlrl/boosting/prediction/transformation_probability_output_wise.cpp',
    'src/mlrl/boosting/rule_evaluation/head_type_auto.cpp',
    'src/mlrl/boosting/rule_evaluation/head_type_complete.cpp',
    'src/mlrl/boosting/rule_evaluation/head_type_partial_dynamic.cpp',
    'src/mlrl/boosting/rule_evaluation/head_type_partial_fixed.cpp',
    'src/mlrl/boosting/rule_evaluation/head_type_single.cpp',
    'src/mlrl/boosting/rule_evaluation/regularization_manual.cpp',
    'src/mlrl/boosting/rule_evaluation/regularization_no.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_complete_binned.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_dynamic_binned.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_partial_fixed_binned.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_single.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_complete_binned.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_dynamic_binned.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed.cpp',
    'src/mlrl/boosting/rule_evaluation/rule_evaluation_non_decomposable_partial_fixed_binned.cpp',
    'src/mlrl/boosting/rule_model_assemblage/default_rule_auto.cpp',
    'src/mlrl/boosting/sampling/partition_sampling_auto.cpp',
    'src/mlrl/boosting/statistics/statistic_format_auto.cpp',
    'src/mlrl/boosting/statistics/statistic_format_dense.cpp',
    'src/mlrl/boosting/statistics/statistic_format_sparse.cpp',
    'src/mlrl/boosting/statistics/statistics_provider_decomposable_dense.cpp',
    'src/mlrl/boosting/statistics/statistics_provider_decomposable_sparse.cpp',
    'src/mlrl/boosting/statistics/statistics_provider_non_decomposable_dense.cpp',
    'src/mlrl/boosting/util/blas.cpp',
    'src/mlrl/boosting/util/lapack.cpp',
    'src/mlrl/boosting/info.cpp',
    'src/mlrl/boosting/learner_boomer_classifier.cpp'
]

# Test files
test_files = [
    'test/mlrl/boosting/info.cpp'
]

# Dependencies
common_project = subproject('common')
common_dep = common_project.get_variable('common_dep')

dependencies = [
    common_dep
]

# Directory containing public headers
include_dir = include_directories('include')

# Directory into which the library should be installed
install_root = common_project.get_variable('install_root')
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Obtain information about the library
lib_name = 'mlrl' + meson.project_name()
version = common_project.get_variable('version')
target_architecture = common_project.get_variable('target_architecture')

# Set configuration options
configuration = configuration_data()
configuration.set('mlrlboosting_library_name', 'lib' + lib_name)
configuration.set('mlrlboosting_library_version', version)
configuration.set('mlrlboosting_target_architecture', target_architecture)
configure_file(input : 'include/mlrl/boosting/config.hpp.in', output : 'config.hpp', configuration: configuration)

# Library declaration
cpp_args = common_project.get_variable('cpp_args')
link_args = common_project.get_variable('link_args')

if host_machine.system() == 'windows'
    cpp_args += '-DMLRLBOOSTING_EXPORTS'
endif

boosting_lib = library(lib_name, source_files, include_directories : include_dir, dependencies : dependencies,
                       cpp_args : cpp_args, link_args : link_args, version : version, install : true,
                       install_dir : install_dir)

# Test declaration
gtest_test_dep = common_project.get_variable('gtest_test_dep')

if gtest_test_dep.found()
    common_test_dep = common_project.get_variable('common_test_dep')

    test_dependencies = [
        gtest_test_dep,
        common_test_dep,
        declare_dependency(include_directories : [include_dir, include_directories('src')], link_with : boosting_lib)
    ]

    test_name = lib_name + 'tests'
    boosting_tests = executable(test_name, test_files, dependencies : test_dependencies)
    test(test_name, boosting_tests)
endif
