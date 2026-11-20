"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Tests for saving BOOMER learners to HDF5 files and loading them from such files.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlrl.boosting.learners import BoomerClassifier, BoomerRegressor
from mlrl.common import load_model
from mlrl.common.cython.model_file import ModelFileReader, ModelFileWriter
from mlrl.common.learners import load_model as load_model_from_learners
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from ..common.pickling import assert_identical_learners, create_classification_dataset, create_regression_dataset


def _save_and_load(learner, path: Path):
    learner.save_model(path)
    return load_model(path)


@pytest.mark.boosting
@pytest.mark.classification
class TestBoomerClassifierSaveLoad:
    """
    Tests for saving fitted instances of the class `BoomerClassifier` and loading them.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'partial-fixed'}, id='partial-heads'),
            pytest.param({'head_type': 'complete', 'loss': 'logistic-non-decomposable'}, id='complete-heads'),
            pytest.param({'binary_predictor': 'example-wise'}, id='binary-predictor-example-wise'),
            pytest.param(
                {
                    'binary_predictor': 'gfm',
                    'probability_predictor': 'marginalized',
                    'marginal_probability_calibration': 'isotonic',
                    'joint_probability_calibration': 'isotonic',
                },
                id='gfm-isotonic-calibration',
            ),
            pytest.param({'prediction_format': 'sparse'}, id='sparse-predictions'),
            pytest.param(
                {'feature_binning': 'equal-width', 'shrinkage': 0.5, 'statistic_type': '32-bit', 'random_state': 7},
                id='non-default-hyperparameters',
            ),
            pytest.param({'shrinkage': None}, id='parameter-set-to-none'),
        ],
    )
    def test_save_load(self, parameters: dict[str, Any], tmp_path: Path):
        x, y, ordinal_feature_indices, nominal_feature_indices = create_classification_dataset()
        learner = BoomerClassifier(max_rules=15, random_state=1)
        learner.set_params(**parameters)
        learner.fit(
            x, y, ordinal_feature_indices=ordinal_feature_indices, nominal_feature_indices=nominal_feature_indices
        )
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        assert type(loaded) is BoomerClassifier
        assert loaded.get_params() == learner.get_params()
        assert loaded.n_features_in_ == learner.n_features_in_
        assert loaded.num_outputs_ == learner.num_outputs_
        assert loaded.sparse_predictions_ == learner.sparse_predictions_
        assert loaded.y_dtype_ == learner.y_dtype_
        np.testing.assert_array_equal(loaded.classes_, learner.classes_)
        assert loaded.classes_.dtype == learner.classes_.dtype
        assert_identical_learners(learner, loaded, x)

    def test_sparse_input(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=15, random_state=1).fit(csr_matrix(x), csr_matrix(y))
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        assert_identical_learners(learner, loaded, csr_matrix(x))

    def test_string_labels(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        y_strings = np.where(y[:, 0] == 1, 'positive', 'negative')
        learner = BoomerClassifier(max_rules=15, random_state=1).fit(x, y_strings)
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        np.testing.assert_array_equal(loaded.label_encoder_.classes_, learner.label_encoder_.classes_)
        np.testing.assert_array_equal(loaded.classes_, learner.classes_)
        assert_identical_learners(learner, loaded, x)
        assert set(loaded.predict(x)) <= {'positive', 'negative'}

    def test_constant_labels(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=15, random_state=1).fit(x, np.ones_like(y))
        assert not hasattr(learner, 'model_')
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        assert not hasattr(loaded, 'model_')
        assert loaded.constant_prediction_ == learner.constant_prediction_
        assert type(loaded.constant_prediction_) is type(learner.constant_prediction_)
        np.testing.assert_array_equal(loaded.predict(x), learner.predict(x))
        np.testing.assert_array_equal(loaded.predict_proba(x), learner.predict_proba(x))

    def test_feature_names(self, tmp_path: Path):
        pandas = pytest.importorskip('pandas')
        x, y, _, _ = create_classification_dataset()
        x_frame = pandas.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        learner = BoomerClassifier(max_rules=15, random_state=1).fit(x_frame, y)
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        # Feature names are only stored if scikit-learn has recorded them when fitting the learner
        assert hasattr(loaded, 'feature_names_in_') == hasattr(learner, 'feature_names_in_')

        if hasattr(learner, 'feature_names_in_'):
            np.testing.assert_array_equal(loaded.feature_names_in_, learner.feature_names_in_)

        np.testing.assert_array_equal(loaded.predict(x_frame), learner.predict(x_frame))

    def test_feature_names_attribute(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1).fit(x, y)
        learner.feature_names_in_ = np.asarray([f'feature_{i}' for i in range(x.shape[1])], dtype=object)
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        np.testing.assert_array_equal(loaded.feature_names_in_, learner.feature_names_in_)
        assert loaded.feature_names_in_.dtype == object

    def test_file_structure(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1, binary_predictor='gfm').fit(x, y)
        path = tmp_path / 'model.h5'
        learner.save_model(str(path))
        reader = ModelFileReader(path.read_bytes())
        assert reader.get_attribute('learner', 'class') == 'mlrl.boosting.learners.BoomerClassifier'
        assert reader.get_attribute('learner', 'package') == 'mlrl-boomer'
        assert reader.get_attribute('learner', 'package_version')
        assert reader.get_attribute('learner/parameters', 'max_rules') == 5
        assert reader.get_attribute('learner/parameters', 'binary_predictor') == 'gfm'
        assert 'loss' in list(reader.read_array('learner', 'none_parameters'))
        assert reader.get_attribute('learner/fit_state', 'num_outputs') == y.shape[1]
        assert reader.has_rule_model()
        assert reader.has_output_space_info()
        assert reader.has_marginal_probability_calibration_model()
        assert reader.has_joint_probability_calibration_model()

    def test_load_via_class(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        path = tmp_path / 'model.h5'
        BoomerClassifier(max_rules=5, random_state=1).fit(x, y).save_model(path)
        assert type(BoomerClassifier.load(path)) is BoomerClassifier
        assert load_model is load_model_from_learners

        with pytest.raises(TypeError, match='BoomerRegressor'):
            BoomerRegressor.load(path)

    def test_save_unfitted_learner(self, tmp_path: Path):
        with pytest.raises(NotFittedError):
            BoomerClassifier().save_model(tmp_path / 'model.h5')

    def test_save_unsupported_hyperparameter(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1).fit(x, y)
        learner.max_rules = [5]

        with pytest.raises(TypeError, match='max_rules'):
            learner.save_model(tmp_path / 'model.h5')

    @staticmethod
    def __write_file(path: Path, class_name: str | None, parameters: dict[str, Any] | None = None):
        writer = ModelFileWriter()

        if class_name is not None:
            writer.set_attribute('learner', 'class', class_name)

        for name, value in (parameters or {}).items():
            writer.set_attribute('learner/parameters', name, value)

        path.write_bytes(writer.finish())

    def test_load_unknown_hyperparameters(self, tmp_path: Path):
        path = tmp_path / 'model.h5'
        self.__write_file(
            path,
            'mlrl.boosting.learners.BoomerClassifier',
            {'max_rules': 10, 'unknown_parameter': 1, 'another_unknown_parameter': 'value'},
        )

        with pytest.raises(ValueError, match='"another_unknown_parameter", "unknown_parameter"'):
            load_model(path)

    @pytest.mark.parametrize(
        'class_name',
        [
            pytest.param('builtins.dict', id='outside-namespace'),
            pytest.param('os.system', id='function-outside-namespace'),
            pytest.param('mlrl.common.cython.model_file.ModelFileWriter', id='not-a-learner'),
            pytest.param('mlrl.boosting.learners.DoesNotExist', id='missing-class'),
            pytest.param('mlrl.does_not_exist.Learner', id='missing-module'),
        ],
    )
    def test_load_unsupported_class(self, class_name: str, tmp_path: Path):
        path = tmp_path / 'model.h5'
        self.__write_file(path, class_name)

        with pytest.raises(ValueError, match='unsupported type'):
            load_model(path)

    def test_load_file_without_learner(self, tmp_path: Path):
        path = tmp_path / 'model.h5'
        self.__write_file(path, class_name=None)

        with pytest.raises(ValueError, match='does not contain a learner'):
            load_model(path)

    def test_load_invalid_file(self, tmp_path: Path):
        path = tmp_path / 'model.h5'
        path.write_bytes(b'this is not a model file')

        with pytest.raises(RuntimeError):
            load_model(path)


@pytest.mark.boosting
@pytest.mark.regression
class TestBoomerRegressorSaveLoad:
    """
    Tests for saving fitted instances of the class `BoomerRegressor` and loading them.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'partial-fixed'}, id='partial-heads'),
            pytest.param({'statistic_type': '32-bit', 'shrinkage': 0.1}, id='non-default-hyperparameters'),
        ],
    )
    def test_save_load(self, parameters: dict[str, Any], tmp_path: Path):
        x, y = create_regression_dataset()
        learner = BoomerRegressor(max_rules=15, random_state=1, **parameters).fit(x, y)
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        assert type(loaded) is BoomerRegressor
        assert loaded.get_params() == learner.get_params()
        assert_identical_learners(learner, loaded, x)

    def test_float32_targets(self, tmp_path: Path):
        x, y = create_regression_dataset()
        learner = BoomerRegressor(max_rules=15, random_state=1).fit(x, y.astype(np.float32))
        loaded = _save_and_load(learner, tmp_path / 'model.h5')
        assert loaded.y_dtype_ == np.float32
        assert_identical_learners(learner, loaded, x)
