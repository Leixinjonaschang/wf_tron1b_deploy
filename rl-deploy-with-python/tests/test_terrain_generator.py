"""Regression tests for the WF_TRON1B offline terrain generator."""

import importlib.util
import xml.etree.ElementTree as xml_et
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "utils" / "terrain_tool" / "terrain_generator.py"
SCENE_PATH = (
    REPO_ROOT
    / "pointfoot-mujoco-sim"
    / "robot-description"
    / "pointfoot"
    / "WF_TRON1B"
    / "xml"
    / "scene_terrain.xml"
)


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("wf_tron1b_terrain_generator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_terrain_geoms_are_depth_visible(scene_path: Path) -> None:
    root = xml_et.parse(scene_path).getroot()
    geoms = root.findall("./worldbody/geom")
    assert geoms
    for geo in geoms:
        assert geo.attrib["group"] == "0"
        assert geo.attrib["contype"] == "1"
        assert geo.attrib["conaffinity"] == "1"
        assert geo.attrib["condim"] == "3"
        assert float(geo.attrib["rgba"].split()[3]) == 1.0


def test_default_course_generation_stamps_every_terrain_geom(tmp_path):
    generator_module = _load_generator_module()
    base_scene = tmp_path / "base_scene.xml"
    output_scene = tmp_path / "scene_terrain.xml"
    base_scene.write_text("<mujoco><asset/><worldbody/></mujoco>", encoding="utf-8")

    generator = generator_module.TerrainGenerator(base_scene, output_scene)
    generator_module.add_default_course(generator)
    generator.Save()

    _assert_terrain_geoms_are_depth_visible(output_scene)


def test_committed_terrain_scene_loads_in_mujoco():
    _assert_terrain_geoms_are_depth_visible(SCENE_PATH)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    assert model.ngeom > 1
