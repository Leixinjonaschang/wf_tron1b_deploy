"""Regression tests for the WF_TRON1B offline terrain generator."""

import importlib.util
import xml.etree.ElementTree as xml_et
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "utils" / "terrain_tool" / "terrain_generator.py"
XML_DIR = (
    REPO_ROOT
    / "pointfoot-mujoco-sim"
    / "robot-description"
    / "pointfoot"
    / "WF_TRON1B"
    / "xml"
)
ROBOT_XML = XML_DIR / "robot.xml"
SCENE_EXPECTATIONS = {
    "scene_stairs.xml": (4, {"box"}),
    "scene_slope.xml": (1, {"box"}),
    "scene_rough_ground.xml": (42, {"box"}),
    "scene_obstacle.xml": (1, {"cylinder"}),
    "scene_terrain.xml": (48, {"box", "cylinder"}),
}


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("wf_tron1b_terrain_generator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terrain_geoms(scene_path: Path):
    root = xml_et.parse(scene_path).getroot()
    geoms = root.findall("./worldbody/geom")
    assert geoms
    return geoms


def _assert_terrain_geoms_are_depth_visible(scene_path: Path) -> None:
    geoms = _terrain_geoms(scene_path)
    for geo in geoms:
        assert geo.attrib["group"] == "0"
        assert geo.attrib["contype"] == "1"
        assert geo.attrib["conaffinity"] == "1"
        assert geo.attrib["condim"] == "3"
        assert float(geo.attrib["rgba"].split()[3]) == 1.0


def test_all_scene_generation_stamps_every_terrain_geom(tmp_path):
    generator_module = _load_generator_module()
    base_scene = tmp_path / "base_scene.xml"
    base_scene.write_text("<mujoco><asset/><worldbody/></mujoco>", encoding="utf-8")

    generated_scenes = generator_module.generate_all_scenes(base_scene, tmp_path)

    assert {scene.name for scene in generated_scenes} == set(SCENE_EXPECTATIONS)
    for output_scene in generated_scenes:
        expected_count, expected_types = SCENE_EXPECTATIONS[output_scene.name]
        geoms = _terrain_geoms(output_scene)
        assert len(geoms) == expected_count
        assert {geo.attrib["type"] for geo in geoms} == expected_types
        _assert_terrain_geoms_are_depth_visible(output_scene)


def test_committed_terrain_scenes_load_in_mujoco():
    for scene_name, (expected_count, expected_types) in SCENE_EXPECTATIONS.items():
        scene_path = XML_DIR / scene_name
        geoms = _terrain_geoms(scene_path)
        assert len(geoms) == expected_count
        assert {geo.attrib["type"] for geo in geoms} == expected_types
        _assert_terrain_geoms_are_depth_visible(scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        assert model.ngeom > expected_count


def test_robot_floor_uses_skeleton_reference_colours():
    root = xml_et.parse(ROBOT_XML).getroot()
    texture = next(texture for texture in root.findall("./asset/texture") if texture.attrib.get("name") == "texplane")
    floor = next(geom for geom in root.findall("./worldbody/geom") if geom.attrib.get("name") == "floor")

    assert texture.attrib["rgb1"] == "1 1 1"
    assert texture.attrib["rgb2"] == "0.85 0.85 0.85"
    assert floor.attrib["rgba"] == "1 1 1 1"
