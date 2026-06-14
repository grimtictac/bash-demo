from __future__ import annotations

import pytest

from app import InferenceService, PredictionRequest


class TestValidateRequest:
    def test_non_dict_body(self, svc):
        with pytest.raises(ValueError, match="JSON object"):
            svc.validate_request("not a dict")

    def test_missing_records_key(self, svc):
        with pytest.raises(ValueError, match="'records'"):
            svc.validate_request({})

    def test_empty_records_list(self, svc):
        with pytest.raises(ValueError, match="non-empty"):
            svc.validate_request({"records": []})

    def test_records_not_a_list(self, svc):
        with pytest.raises(ValueError, match="'records'"):
            svc.validate_request({"records": "oops"})

    def test_record_not_a_dict(self, svc):
        with pytest.raises(ValueError, match=r"records\[0\]"):
            svc.validate_request({"records": ["not a dict"]})

    def test_valid_minimal(self, svc):
        req = svc.validate_request({"records": [{"a": 1}]})
        assert req.records == [{"a": 1}]
        assert req.request_id is None

    def test_valid_with_request_id(self, svc):
        req = svc.validate_request({"records": [{"a": 1}], "request_id": "abc-123"})
        assert req.request_id == "abc-123"

    def test_invalid_request_id_type(self, svc):
        with pytest.raises(ValueError, match="request_id"):
            svc.validate_request({"records": [{"a": 1}], "request_id": 42})

    def test_records_are_deep_copied(self, svc):
        """Mutations to the original payload must not affect the parsed request."""
        original = {"nested": {"x": 1}}
        req = svc.validate_request({"records": [original]})
        original["nested"]["x"] = 999
        assert req.records[0]["nested"]["x"] == 1

    def test_multiple_records(self, svc):
        req = svc.validate_request({"records": [{"a": 1}, {"b": 2}, {"c": 3}]})
        assert len(req.records) == 3
