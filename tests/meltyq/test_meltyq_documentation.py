import hashlib
import json
from types import SimpleNamespace

import pytest

from docs.meltyq import (
    build_meltyq_figure3_raster_comparison as figure3_docs,
)
from docs.meltyq import meltyq_figure8_document_support as figure8_docs


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _figure3_args(tmp_path):
    return SimpleNamespace(
        raster_reference=figure3_docs.DEFAULT_RASTER_REFERENCE,
        model_output=figure3_docs.DEFAULT_ARCHIVED_MODEL,
        archived_model=figure3_docs.DEFAULT_ARCHIVED_MODEL,
        metadata=figure3_docs.DEFAULT_ARCHIVED_METADATA,
        archived_metadata=figure3_docs.DEFAULT_ARCHIVED_METADATA,
        public_config=figure3_docs.DEFAULT_PUBLIC_CONFIG,
        summary=tmp_path / "summary.json",
        ja_figure=tmp_path / "comparison_ja.png",
        en_figure=tmp_path / "comparison_en.png",
    )


def test_figure3_archived_metadata_identifies_the_exact_model(tmp_path):
    model = figure3_docs.load_model(figure3_docs.DEFAULT_ARCHIVED_MODEL)
    metadata = figure3_docs.load_json(
        figure3_docs.DEFAULT_ARCHIVED_METADATA
    )
    figure3_docs.validate_archived_pair(
        metadata,
        model,
        figure3_docs.DEFAULT_ARCHIVED_MODEL,
    )

    changed_model_path = tmp_path / "model.csv"
    changed_model_path.write_bytes(
        figure3_docs.DEFAULT_ARCHIVED_MODEL.read_bytes() + b"\n"
    )
    changed_model = figure3_docs.load_model(changed_model_path)
    with pytest.raises(ValueError, match="does not match"):
        figure3_docs.validate_archived_pair(
            metadata,
            changed_model,
            changed_model_path,
        )


def test_figure3_archive_refresh_requires_model_and_metadata_together(tmp_path):
    args = _figure3_args(tmp_path)
    args.model_output = tmp_path / "live-model.csv"
    with pytest.raises(ValueError, match="together"):
        figure3_docs.build(args)


def test_figure3_default_document_builder_is_deterministic(tmp_path):
    args = _figure3_args(tmp_path)
    figure3_docs.build(args)
    first = {
        path.name: _sha256(path)
        for path in (args.summary, args.ja_figure, args.en_figure)
    }
    figure3_docs.build(args)
    second = {
        path.name: _sha256(path)
        for path in (args.summary, args.ja_figure, args.en_figure)
    }
    assert first == second

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    sources = summary["inputs"]["source_outputs"]
    assert sources["model_spectra_csv"]["path"].startswith(
        "docs/meltyq/data/"
    )
    assert sources["metadata_json"]["path"].startswith(
        "docs/meltyq/data/"
    )


def test_figure8_fingerprint_tracks_exopie_data_assets(monkeypatch):
    states = {
        "exopie": {
            "python_source_files": 1,
            "python_source_inventory_sha256": "a" * 64,
            "data_files": 1,
            "data_inventory_sha256": "b" * 64,
        }
    }
    monkeypatch.setattr(
        figure8_docs,
        "runtime_versions",
        lambda: {"python": "test"},
    )
    monkeypatch.setattr(
        figure8_docs,
        "runtime_source_states",
        lambda: states,
    )
    first = figure8_docs._calculation_fingerprint(
        figure3_docs.REPOSITORY_ROOT,
        basis_aligned=False,
    )
    states["exopie"]["data_inventory_sha256"] = "c" * 64
    second = figure8_docs._calculation_fingerprint(
        figure3_docs.REPOSITORY_ROOT,
        basis_aligned=False,
    )
    assert first != second


def test_figure8_cache_rejects_changed_artifacts(tmp_path):
    fingerprint = "f" * 64
    for filename in figure8_docs.DOCUMENT_CACHE_ARTIFACTS:
        (tmp_path / filename).write_bytes(filename.encode())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "document_cache": {
                    "schema": figure8_docs.DOCUMENT_CACHE_SCHEMA_VERSION,
                    "calculation_fingerprint": fingerprint,
                    "artifacts": figure8_docs._artifact_records(tmp_path),
                }
            }
        ),
        encoding="utf-8",
    )

    assert figure8_docs._cache_matches(metadata_path, fingerprint)
    (tmp_path / "results.csv").write_bytes(b"changed")
    assert not figure8_docs._cache_matches(metadata_path, fingerprint)
