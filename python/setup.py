import sys

import numpy
import setuptools
from Cython.Build import cythonize

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = False

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

# The compiler/linker argument to enable OpenMP support
COMPILE_FLAG_OPEN_MP = '/openmp' if sys.platform.startswith('win') else '-fopenmp'

sources = [
    '**/*.pyx',
    'common/cpp/data/matrix_dense.cpp',
    'common/cpp/data/matrix_dok_binary.cpp',
    'common/cpp/data/vector_dense.cpp',
    'common/cpp/data/vector_dok_binary.cpp',
    'common/cpp/data/vector_sparse_array.cpp',
    'common/cpp/data/vector_sparse_list_binary.cpp',
    'common/cpp/data/vector_mapping_dense.cpp',
    'common/cpp/data/view_c_contiguous.cpp',
    'common/cpp/data/view_csc.cpp',
    'common/cpp/data/view_csr.cpp',
    'common/cpp/data/view_fortran_contiguous.cpp',
    'common/cpp/indices/index_iterator.cpp',
    'common/cpp/indices/index_vector_full.cpp',
    'common/cpp/indices/index_vector_partial.cpp',
    'common/cpp/input/feature_matrix_csc.cpp',
    'common/cpp/input/feature_matrix_fortran_contiguous.cpp',
    'common/cpp/input/feature_vector.cpp',
    'common/cpp/input/label_matrix_c_contiguous.cpp',
    'common/cpp/input/label_matrix_dok.cpp',
    'common/cpp/input/nominal_feature_mask_dok.cpp',
    'common/cpp/model/body_conjunctive.cpp',
    'common/cpp/model/body_empty.cpp',
    'common/cpp/model/condition.cpp',
    'common/cpp/model/condition_list.cpp',
    'common/cpp/model/head_full.cpp',
    'common/cpp/model/head_partial.cpp',
    'common/cpp/model/rule.cpp',
    'common/cpp/model/rule_model.cpp',
    'common/cpp/sampling/random.cpp',
    'common/cpp/sampling/feature_sampling_no.cpp',
    'common/cpp/sampling/feature_sampling_random.cpp',
    'common/cpp/sampling/instance_sampling_bagging.cpp',
    'common/cpp/sampling/instance_sampling_no.cpp',
    'common/cpp/sampling/instance_sampling_random.cpp',
    'common/cpp/sampling/label_sampling_no.cpp',
    'common/cpp/sampling/label_sampling_random.cpp',
    'common/cpp/sampling/weight_vector_dense.cpp',
    'common/cpp/sampling/weight_vector_equal.cpp',
    'common/cpp/stopping/stopping_criterion_size.cpp',
    'common/cpp/stopping/stopping_criterion_time.cpp',
    'common/cpp/head_refinement/head_refinement_full.cpp',
    'common/cpp/head_refinement/head_refinement_single.cpp',
    'common/cpp/head_refinement/prediction.cpp',
    'common/cpp/head_refinement/prediction_evaluated.cpp',
    'common/cpp/head_refinement/prediction_full.cpp',
    'common/cpp/head_refinement/prediction_partial.cpp',
    'common/cpp/post_processing/post_processor_no.cpp',
    'common/cpp/pruning/pruning_irep.cpp',
    'common/cpp/pruning/pruning_no.cpp',
    'common/cpp/rule_evaluation/score_vector_dense.cpp',
    'common/cpp/rule_evaluation/score_vector_label_wise_dense.cpp',
    'common/cpp/rule_induction/rule_induction_top_down.cpp',
    'common/cpp/rule_induction/rule_model_induction_sequential.cpp',
    'common/cpp/rule_refinement/rule_refinement.cpp',
    'common/cpp/rule_refinement/rule_refinement_exact.cpp',
    'common/cpp/rule_refinement/rule_refinement_approximate.cpp',
    'common/cpp/binning/bin_vector.cpp',
    'common/cpp/binning/feature_binning_equal_frequency.cpp',
    'common/cpp/binning/feature_binning_equal_width.cpp',
    'common/cpp/binning/feature_binning_nominal.cpp',
    'common/cpp/thresholds/coverage_mask.cpp',
    'common/cpp/thresholds/thresholds_exact.cpp',
    'common/cpp/thresholds/thresholds_approximate.cpp',
    'boosting/cpp/data/matrix_dense_numeric.cpp',
    'boosting/cpp/data/matrix_dense_label_wise.cpp',
    'boosting/cpp/data/matrix_dense_example_wise.cpp',
    'boosting/cpp/data/vector_dense_label_wise.cpp',
    'boosting/cpp/data/vector_dense_example_wise.cpp',
    'boosting/cpp/losses/loss_label_wise_logistic.cpp',
    'boosting/cpp/losses/loss_label_wise_squared_error.cpp',
    'boosting/cpp/losses/loss_label_wise_squared_hinge.cpp',
    'boosting/cpp/losses/loss_example_wise_logistic.cpp',
    'boosting/cpp/math/blas.cpp',
    'boosting/cpp/math/lapack.cpp',
    'boosting/cpp/model/rule_list.cpp',
    'boosting/cpp/output/predictor_classification_label_wise.cpp',
    'boosting/cpp/output/predictor_classification_example_wise.cpp',
    'boosting/cpp/post_processing/shrinkage_constant.cpp',
    'boosting/cpp/rule_evaluation/rule_evaluation_example_wise_regularized.cpp',
    'boosting/cpp/rule_evaluation/rule_evaluation_label_wise_regularized.cpp',
    'boosting/cpp/statistics/statistics_label_wise_dense.cpp',
    'boosting/cpp/statistics/statistics_label_wise_provider.cpp',
    'boosting/cpp/statistics/statistics_example_wise_dense.cpp',
    'boosting/cpp/statistics/statistics_example_wise_provider.cpp',
    'seco/cpp/head_refinement/head_refinement_partial.cpp',
    'seco/cpp/head_refinement/lift_function_peak.cpp',
    'seco/cpp/heuristics/heuristic_f_measure.cpp',
    'seco/cpp/heuristics/heuristic_hamming_loss.cpp',
    'seco/cpp/heuristics/heuristic_m_estimate.cpp',
    'seco/cpp/heuristics/heuristic_precision.cpp',
    'seco/cpp/heuristics/heuristic_recall.cpp',
    'seco/cpp/heuristics/heuristic_wra.cpp',
    'seco/cpp/model/decision_list.cpp',
    'seco/cpp/output/predictor_classification_label_wise.cpp',
    'seco/cpp/rule_evaluation/rule_evaluation_label_wise_heuristic.cpp',
    'seco/cpp/statistics/statistics_label_wise_dense.cpp',
    'seco/cpp/statistics/statistics_label_wise_provider.cpp',
    'seco/cpp/stopping/stopping_criterion_coverage.cpp'
]

extensions = [
    setuptools.Extension(name='*', sources=sources, language='c++', extra_compile_args=[COMPILE_FLAG_OPEN_MP],
                         extra_link_args=[COMPILE_FLAG_OPEN_MP],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

setuptools.setup(
    name='boomer',
    version='0.4.0',
    description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
    url='https://github.com/mrapp-ke/Boomer',
    author='Michael Rapp',
    author_email='mrapp@ke.tu-darmstadt.de',
    license='MIT',
    packages=['boomer'],
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'Cython>=0.29.0',
        'scikit-learn>=0.23.0',
        'liac-arff>=2.5.0',
        'requests>=2.25.0'
    ],
    python_requires='>=3.7',
    ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()])
