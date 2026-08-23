from pathlib import Path

import pytest

from orcs.assets import objects

assets = pytest.importorskip("assets")


def test_object_xml_prefers_package_resolver(monkeypatch, tmp_path):
    generated = tmp_path / "generated/suitcase_cvx_dcmp.xml"
    generated.parent.mkdir()
    generated.touch()
    calls = []

    def resolve(*parts: str) -> Path:
        calls.append(parts)
        return generated

    monkeypatch.setattr(assets, "resolve_asset", resolve)
    monkeypatch.setattr(objects, "assets_source", lambda: tmp_path / "tracked")

    assert objects._object_xml("suitcase", "cvx_dcmp") == generated
    assert calls == [("omomo_objects", "suitcase", "suitcase_cvx_dcmp.xml")]


def test_object_xml_supports_legacy_tracked_layout(monkeypatch, tmp_path):
    tracked = tmp_path / "tracked"
    xml = tracked / "custom_objects/tire/tire_cvx_hull.xml"
    xml.parent.mkdir(parents=True)
    xml.touch()

    monkeypatch.delattr(assets, "resolve_asset")
    monkeypatch.setattr(objects, "assets_source", lambda: tracked)

    assert objects._object_xml("tire", "cvx_hull") == xml
