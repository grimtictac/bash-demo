from __future__ import annotations

import pytest

from app import PredictionError


class TestValidateResponse:
    def test_wrong_count(self, svc):
        with pytest.raises(PredictionError, match="count"):
            svc.validate_response([1, 2], 3)

    def test_not_a_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response(42, 1)

    def test_string_rejected_as_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response("abc", 3)

    def test_bytes_rejected_as_sequence(self, svc):
        with pytest.raises(PredictionError):
            svc.validate_response(b"abc", 3)

    def test_none_prediction(self, svc):
        with pytest.raises(PredictionError, match="None"):
            svc.validate_response([None], 1)

    def test_nan_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("nan")}], 1)

    def test_positive_inf_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("inf")}], 1)

    def test_negative_inf_in_prediction_dict(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"score": float("-inf")}], 1)

    def test_nan_nested_in_list(self, svc):
        with pytest.raises(PredictionError, match="non-finite"):
            svc.validate_response([{"scores": [1.0, float("nan")]}], 1)

    def test_valid_predictions_pass_through(self, svc):
        result = svc.validate_response([{"score": 1.0}, {"score": 2.0}], 2)
        assert result == [{"score": 1.0}, {"score": 2.0}]

    def test_integer_scores_are_accepted(self, svc):
        result = svc.validate_response([{"score": 1}], 1)
        assert result == [{"score": 1}]
