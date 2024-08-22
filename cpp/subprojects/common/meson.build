project('common', 'cpp')

# Source files
source_files = [
    'src/mlrl/common/data/vector_bit.cpp',
    'src/mlrl/common/iterator/iterator_index.cpp',
    'src/mlrl/common/indices/index_vector_complete.cpp',
    'src/mlrl/common/indices/index_vector_partial.cpp',
    'src/mlrl/common/input/feature_binning_equal_frequency.cpp',
    'src/mlrl/common/input/feature_binning_equal_width.cpp',
    'src/mlrl/common/input/feature_binning_no.cpp',
    'src/mlrl/common/input/feature_info_equal.cpp',
    'src/mlrl/common/input/feature_info_mixed.cpp',
    'src/mlrl/common/input/feature_matrix_c_contiguous.cpp',
    'src/mlrl/common/input/feature_matrix_csc.cpp',
    'src/mlrl/common/input/feature_matrix_csr.cpp',
    'src/mlrl/common/input/feature_matrix_fortran_contiguous.cpp',
    'src/mlrl/common/input/feature_type_nominal.cpp',
    'src/mlrl/common/input/feature_type_numerical.cpp',
    'src/mlrl/common/input/feature_type_ordinal.cpp',
    'src/mlrl/common/input/feature_vector_binned.cpp',
    'src/mlrl/common/input/feature_vector_equal.cpp',
    'src/mlrl/common/input/feature_vector_nominal.cpp',
    'src/mlrl/common/input/feature_vector_numerical.cpp',
    'src/mlrl/common/input/label_matrix_c_contiguous.cpp',
    'src/mlrl/common/input/label_matrix_csr.cpp',
    'src/mlrl/common/input/regression_matrix_c_contiguous.cpp',
    'src/mlrl/common/input/regression_matrix_csr.cpp',
    'src/mlrl/common/model/body_conjunctive.cpp',
    'src/mlrl/common/model/body_empty.cpp',
    'src/mlrl/common/model/condition_list.cpp',
    'src/mlrl/common/model/head_complete.cpp',
    'src/mlrl/common/model/head_partial.cpp',
    'src/mlrl/common/model/rule_list.cpp',
    'src/mlrl/common/multi_threading/multi_threading_manual.cpp',
    'src/mlrl/common/multi_threading/multi_threading_no.cpp',
    'src/mlrl/common/post_optimization/model_builder_intermediate.cpp',
    'src/mlrl/common/post_optimization/post_optimization_no.cpp',
    'src/mlrl/common/post_optimization/post_optimization_phase_list.cpp',
    'src/mlrl/common/post_optimization/post_optimization_sequential.cpp',
    'src/mlrl/common/post_optimization/post_optimization_unused_rule_removal.cpp',
    'src/mlrl/common/post_processing/post_processor_no.cpp',
    'src/mlrl/common/prediction/label_vector_set.cpp',
    'src/mlrl/common/prediction/output_space_info_no.cpp',
    'src/mlrl/common/prediction/prediction_matrix_dense.cpp',
    'src/mlrl/common/prediction/prediction_matrix_sparse_binary.cpp',
    'src/mlrl/common/prediction/predictor_binary_no.cpp',
    'src/mlrl/common/prediction/predictor_probability_no.cpp',
    'src/mlrl/common/prediction/predictor_score_no.cpp',
    'src/mlrl/common/prediction/probability_calibration_isotonic.cpp',
    'src/mlrl/common/prediction/probability_calibration_no.cpp',
    'src/mlrl/common/random/rng.cpp',
    'src/mlrl/common/rule_evaluation/score_vector_binned_dense.cpp',
    'src/mlrl/common/rule_evaluation/score_vector_dense.cpp',
    'src/mlrl/common/rule_induction/rule_induction_top_down_beam_search.cpp',
    'src/mlrl/common/rule_induction/rule_induction_top_down_greedy.cpp',
    'src/mlrl/common/rule_model_assemblage/default_rule.cpp',
    'src/mlrl/common/rule_model_assemblage/rule_model_assemblage_sequential.cpp',
    'src/mlrl/common/rule_pruning/rule_pruning_irep.cpp',
    'src/mlrl/common/rule_pruning/rule_pruning_no.cpp',
    'src/mlrl/common/rule_refinement/coverage_mask.cpp',
    'src/mlrl/common/rule_refinement/feature_based_search.cpp',
    'src/mlrl/common/rule_refinement/feature_space_tabular.cpp',
    'src/mlrl/common/rule_refinement/prediction_complete.cpp',
    'src/mlrl/common/rule_refinement/prediction_partial.cpp',
    'src/mlrl/common/rule_refinement/refinement_comparator_fixed.cpp',
    'src/mlrl/common/rule_refinement/refinement_comparator_single.cpp',
    'src/mlrl/common/rule_refinement/rule_refinement_feature_based.cpp',
    'src/mlrl/common/rule_refinement/score_processor.cpp',
    'src/mlrl/common/sampling/feature_sampling_no.cpp',
    'src/mlrl/common/sampling/feature_sampling_predefined.cpp',
    'src/mlrl/common/sampling/feature_sampling_without_replacement.cpp',
    'src/mlrl/common/sampling/instance_sampling_no.cpp',
    'src/mlrl/common/sampling/instance_sampling_stratified_example_wise.cpp',
    'src/mlrl/common/sampling/instance_sampling_stratified_output_wise.cpp',
    'src/mlrl/common/sampling/instance_sampling_with_replacement.cpp',
    'src/mlrl/common/sampling/instance_sampling_without_replacement.cpp',
    'src/mlrl/common/sampling/output_sampling_no.cpp',
    'src/mlrl/common/sampling/output_sampling_round_robin.cpp',
    'src/mlrl/common/sampling/output_sampling_without_replacement.cpp',
    'src/mlrl/common/sampling/partition_bi.cpp',
    'src/mlrl/common/sampling/partition_sampling_bi_random.cpp',
    'src/mlrl/common/sampling/partition_sampling_bi_stratified_example_wise.cpp',
    'src/mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.cpp',
    'src/mlrl/common/sampling/partition_sampling_no.cpp',
    'src/mlrl/common/sampling/partition_single.cpp',
    'src/mlrl/common/sampling/stratified_sampling_example_wise.cpp',
    'src/mlrl/common/sampling/stratified_sampling_output_wise.cpp',
    'src/mlrl/common/sampling/weight_vector_bit.cpp',
    'src/mlrl/common/sampling/weight_vector_dense.cpp',
    'src/mlrl/common/sampling/weight_vector_equal.cpp',
    'src/mlrl/common/sampling/weight_vector_out_of_sample.cpp',
    'src/mlrl/common/stopping/global_pruning_no.cpp',
    'src/mlrl/common/stopping/global_pruning_post.cpp',
    'src/mlrl/common/stopping/global_pruning_pre.cpp',
    'src/mlrl/common/stopping/stopping_criterion_list.cpp',
    'src/mlrl/common/stopping/stopping_criterion_no.cpp',
    'src/mlrl/common/stopping/stopping_criterion_size.cpp',
    'src/mlrl/common/stopping/stopping_criterion_time.cpp',
    'src/mlrl/common/info.cpp'
]

# Test files
test_files = [
    'test/mlrl/common/data/array.cpp',
    'test/mlrl/common/data/vector_bit.cpp',
    'test/mlrl/common/data/vector_dense.cpp',
    'test/mlrl/common/input/feature_binning_equal_frequency.cpp',
    'test/mlrl/common/input/feature_binning_equal_width.cpp',
    'test/mlrl/common/input/feature_type_nominal.cpp',
    'test/mlrl/common/input/feature_type_numerical.cpp',
    'test/mlrl/common/input/feature_type_ordinal.cpp',
    'test/mlrl/common/input/feature_vector_decorator_binary.cpp',
    'test/mlrl/common/input/feature_vector_decorator_binned.cpp',
    'test/mlrl/common/input/feature_vector_decorator_nominal.cpp',
    'test/mlrl/common/input/feature_vector_decorator_numerical.cpp',
    'test/mlrl/common/input/feature_vector_decorator_ordinal.cpp',
    'test/mlrl/common/input/feature_vector_equal.cpp',
    'test/mlrl/common/info.cpp'
]

# Directory containing public headers
include_dir = include_directories('include')

# Directory into which the library should be installed
install_root = meson.current_source_dir() / '../../../python/subprojects'
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Obtain information about the library
lib_name = 'mlrl' + meson.project_name()
fs = import('fs')
full_version = fs.read('../../../.version')
version_numbers = full_version.split('.')
library_version = '.'.join([version_numbers[0], version_numbers[1], version_numbers[2]])
target_architecture = host_machine.cpu_family() + '-' + host_machine.system()

# Check build options
multi_threading_support_enabled = get_option('multi_threading_support').enabled()
gpu_support_enabled = get_option('gpu_support').enabled()

# Set configuration options
configuration = configuration_data()
configuration.set('mlrlcommon_library_name', 'lib' + lib_name)
configuration.set('mlrlcommon_library_version', full_version)
configuration.set('mlrlcommon_target_architecture', target_architecture)
configuration.set10('multi_threading_support_enabled', multi_threading_support_enabled)
configuration.set10('gpu_support_enabled', gpu_support_enabled)
configure_file(input : 'include/mlrl/common/config.hpp.in', output : 'config.hpp', configuration: configuration)

# Compiler and linker arguments
cpp_args = []
link_args = []

if host_machine.system() == 'windows'
    cpp_args += '-DMLRLCOMMON_EXPORTS'
    cpp_args += '/wd4250'  # Supress MSVC compiler warning C4250 ("inherits via dominance")

    if multi_threading_support_enabled
        cpp_args += '/openmp'
    endif

    if gpu_support_enabled
        link_args += 'OpenCL.lib'
    endif
elif host_machine.system() == 'darwin'
    cpp_args += '-Xclang'

    if multi_threading_support_enabled
        cpp_args += '-fopenmp'
        link_args += '-lomp'
    endif

    if gpu_support_enabled
        link_args += '-framework'
        link_args += 'OpenCL'
    endif
else
    if multi_threading_support_enabled
        cpp_args += '-fopenmp'
        link_args += '-fopenmp'
    endif

    if gpu_support_enabled
        link_args += '-lOpenCL'
    endif
endif

# Library declaration
common_lib = library(lib_name, source_files, include_directories : include_dir, cpp_args : cpp_args,
                     link_args : link_args, version : library_version, install : true, install_dir : install_dir)
common_static_lib = static_library(lib_name, source_files, include_directories : include_dir, cpp_args : cpp_args,
                                   link_args : link_args)
common_dep = declare_dependency(include_directories : include_dir, link_with : common_static_lib)

# Test declaration
test_support_option = get_option('test_support')
gtest_test_dep = dependency('gtest', main : true, required : test_support_option)

if gtest_test_dep.found()
    common_test_dep = declare_dependency(include_directories : [include_dir, include_directories('src')],
                                         link_with : common_lib)

    test_dependencies = [
        gtest_test_dep,
        common_test_dep
    ]

    test_name = lib_name + 'tests'
    common_tests = executable(test_name, test_files, dependencies : test_dependencies)
    test(test_name, common_tests)
endif
