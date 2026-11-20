"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Tests for the classes `ModelFileWriter` and `ModelFileReader` that are provided by the package "mlrl-common".
"""

import numpy as np
import pytest
from mlrl.boosting.learners import BoomerClassifier
from mlrl.common.cython.model_file import ModelFileReader, ModelFileWriter, get_format_version
from mlrl.common.cython.output_space_info import LabelVectorSet
from mlrl.common.cython.probability_calibration import (
    IsotonicJointProbabilityCalibrationModel,
    IsotonicMarginalProbabilityCalibrationModel,
)
from mlrl.common.cython.rule_model import RuleList

from ..common.pickling import create_classification_dataset


@pytest.mark.boosting
@pytest.mark.classification
class TestModelFile:
    """
    Tests for writing and reading model files.
    """

    def test_format_and_library_version(self):
        reader = ModelFileReader(ModelFileWriter().finish())
        assert reader.get_format_version() == get_format_version() >= 1
        assert reader.get_library_version()

    def test_attributes(self):
        writer = ModelFileWriter()
        writer.set_attribute('learner', 'class', 'mlrl.boosting.BoomerClassifier')
        writer.set_attribute('learner/parameters', 'max_rules', 1000)
        writer.set_attribute('learner/parameters', 'shrinkage', 0.3)
        writer.set_attribute('learner/parameters', 'flag', True)
        writer.set_attribute('learner/parameters', 'numpy_int', np.int32(-7))
        writer.set_attribute('learner/parameters', 'numpy_float', np.float32(0.5))
        writer.set_attribute('learner/parameters', 'unicode', 'äöü €')
        writer.set_attribute('learner/parameters', 'big', 2**62)
        reader = ModelFileReader(writer.finish())

        assert reader.has_group('learner')
        assert reader.has_group('learner/parameters')
        assert not reader.has_group('missing')
        assert reader.get_attribute('learner', 'class') == 'mlrl.boosting.BoomerClassifier'
        assert reader.has_attribute('learner/parameters', 'max_rules')
        assert not reader.has_attribute('learner/parameters', 'missing')
        assert reader.get_attributes('learner/parameters') == {
            'big': 2**62,
            'flag': 1,
            'max_rules': 1000,
            'numpy_float': 0.5,
            'numpy_int': -7,
            'shrinkage': 0.3,
            'unicode': 'äöü €',
        }

    def test_unsupported_attribute_type(self):
        with pytest.raises(TypeError):
            ModelFileWriter().set_attribute('learner', 'value', None)

        with pytest.raises(TypeError):
            ModelFileWriter().set_attribute('learner', 'value', [1, 2])

    def test_arrays(self):
        writer = ModelFileWriter()
        writer.write_array('learner/fit_state', 'ints', np.array([-1, 0, 2**40], dtype=np.int64))
        writer.write_array('learner/fit_state', 'small_ints', np.array([1, 2], dtype=np.uint8))
        writer.write_array('learner/fit_state', 'bools', np.array([True, False]))
        writer.write_array('learner/fit_state', 'floats', [0.5, -1.25])
        writer.write_array('learner/fit_state', 'float32s', np.array([0.5], dtype=np.float32))
        writer.write_array('learner/fit_state', 'strings', np.array(['a', 'äö', '']))
        writer.write_array('learner/fit_state', 'object_strings', np.array(['x', 'y'], dtype=object))
        writer.write_array('learner/fit_state', 'empty_ints', np.array([], dtype=np.int64))
        writer.write_array('learner/fit_state', 'empty_floats', np.array([], dtype=np.float64))
        writer.write_array('learner/fit_state', 'empty_strings', np.array([], dtype=str))
        reader = ModelFileReader(writer.finish())

        def read(name: str) -> np.ndarray:
            assert reader.has_dataset('learner/fit_state', name)
            return reader.read_array('learner/fit_state', name)

        np.testing.assert_array_equal(read('ints'), [-1, 0, 2**40])
        assert read('ints').dtype == np.int64
        np.testing.assert_array_equal(read('small_ints'), [1, 2])
        np.testing.assert_array_equal(read('bools'), [1, 0])
        np.testing.assert_array_equal(read('floats'), [0.5, -1.25])
        assert read('floats').dtype == np.float64
        np.testing.assert_array_equal(read('float32s'), [0.5])
        assert list(read('strings')) == ['a', 'äö', '']
        assert list(read('object_strings')) == ['x', 'y']
        assert read('empty_ints').shape == (0,) and read('empty_ints').dtype == np.int64
        assert read('empty_floats').shape == (0,) and read('empty_floats').dtype == np.float64
        assert read('empty_strings').shape == (0,)
        assert not reader.has_dataset('learner/fit_state', 'missing')

    def test_unsupported_arrays(self):
        with pytest.raises(ValueError):
            ModelFileWriter().write_array('group', 'matrix', np.zeros((2, 2)))

        with pytest.raises(TypeError):
            ModelFileWriter().write_array('group', 'complex', np.array([1j]))

        with pytest.raises(TypeError):
            ModelFileWriter().write_array('group', 'mixed', np.array(['a', 1], dtype=object))

        with pytest.raises(OverflowError):
            ModelFileWriter().write_array('group', 'large', np.array([2**64 - 1], dtype=np.uint64))

    def test_writing_existing_array_fails(self):
        writer = ModelFileWriter()
        writer.write_array('group', 'array', [1])

        with pytest.raises(RuntimeError):
            writer.write_array('group', 'array', [2])

    def test_writer_cannot_be_used_after_finish(self):
        writer = ModelFileWriter()
        writer.finish()

        with pytest.raises(RuntimeError):
            writer.finish()

        with pytest.raises(RuntimeError):
            writer.set_attribute('group', 'name', 1)

    def test_reading_invalid_data_fails(self):
        with pytest.raises(RuntimeError):
            ModelFileReader(b'')

        with pytest.raises(RuntimeError):
            ModelFileReader(b'no model file')

        reader = ModelFileReader(ModelFileWriter().finish())

        with pytest.raises(RuntimeError):
            reader.get_attribute('missing', 'name')

        with pytest.raises(RuntimeError):
            reader.read_array('missing', 'name')

        with pytest.raises(RuntimeError):
            reader.read_rule_model()

    def test_models(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(
            max_rules=10,
            random_state=1,
            binary_predictor='gfm',
            probability_predictor='marginalized',
            marginal_probability_calibration='isotonic',
            joint_probability_calibration='isotonic',
        )
        learner.fit(x, y)

        writer = ModelFileWriter()
        writer.set_attribute('learner', 'class', 'mlrl.boosting.BoomerClassifier')
        writer.write_rule_model(learner.model_)
        writer.write_output_space_info(learner.output_space_info_)
        writer.write_marginal_probability_calibration_model(learner.marginal_probability_calibration_model_)
        writer.write_joint_probability_calibration_model(learner.joint_probability_calibration_model_)
        reader = ModelFileReader(writer.finish())

        assert reader.has_rule_model()
        assert reader.has_output_space_info()
        assert reader.has_marginal_probability_calibration_model()
        assert reader.has_joint_probability_calibration_model()
        rule_model = reader.read_rule_model()
        assert isinstance(rule_model, RuleList)
        assert rule_model.get_num_rules() == learner.model_.get_num_rules()
        assert isinstance(reader.read_output_space_info(), LabelVectorSet)
        assert isinstance(
            reader.read_marginal_probability_calibration_model(), IsotonicMarginalProbabilityCalibrationModel
        )
        assert isinstance(reader.read_joint_probability_calibration_model(), IsotonicJointProbabilityCalibrationModel)

    def test_writing_none_model_fails(self):
        with pytest.raises(TypeError):
            ModelFileWriter().write_rule_model(None)
