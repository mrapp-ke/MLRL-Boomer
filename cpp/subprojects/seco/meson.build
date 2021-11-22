project('seco', 'cpp')

# Source files
source_files = [
    'src/seco/data/matrix_weight_dense.cpp',
    'src/seco/data/vector_confusion_matrix_dense.cpp',
    'src/seco/heuristics/heuristic_accuracy.cpp',
    'src/seco/heuristics/heuristic_f_measure.cpp',
    'src/seco/heuristics/heuristic_laplace.cpp',
    'src/seco/heuristics/heuristic_m_estimate.cpp',
    'src/seco/heuristics/heuristic_precision.cpp',
    'src/seco/heuristics/heuristic_recall.cpp',
    'src/seco/heuristics/heuristic_wra.cpp',
    'src/seco/model/decision_list.cpp',
    'src/seco/output/predictor_classification_label_wise.cpp',
    'src/seco/rule_evaluation/lift_function_peak.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_majority.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_partial.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_single.cpp',
    'src/seco/sampling/instance_sampling_no.cpp',
    'src/seco/sampling/instance_sampling_with_replacement.cpp',
    'src/seco/sampling/instance_sampling_without_replacement.cpp',
    'src/seco/statistics/statistics_provider_factory_label_wise_dense.cpp',
    'src/seco/stopping/stopping_criterion_coverage.cpp'
]

# Dependencies
common_project = subproject('common')
common_dep = common_project.get_variable('common_dep')
openmp_dep = dependency('openmp')

dependencies = [
    common_dep,
    openmp_dep
]

# Directory containing public headers
include_directories = include_directories('include')

# Directory into which the library should be installed
install_root = common_project.get_variable('install_root')
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Library version
version = common_project.get_variable('version')

# Library declaration
seco_lib = library('mlrl' + meson.project_name(), source_files, include_directories : include_directories,
                   dependencies : dependencies, version : version, install : true, install_dir : install_dir)
