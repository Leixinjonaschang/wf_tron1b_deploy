"""Generate depth-visible MuJoCo terrain scenes for WF_TRON1B.

The generated scene includes the existing ``robot.xml`` and appends terrain to
its worldbody. Run this file directly after changing the course definition.
Height-field helpers are optional and only require OpenCV and ``noise`` when
they are called.
"""

from __future__ import annotations

import xml.etree.ElementTree as xml_et
from pathlib import Path

import numpy as np


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
XML_DIR = REPO_ROOT / "pointfoot-mujoco-sim" / "robot-description" / "pointfoot" / "WF_TRON1B" / "xml"
INPUT_SCENE_PATH = TOOL_DIR / "base_scene.xml"
OUTPUT_SCENE_PATH = XML_DIR / "scene_terrain.xml"

DEPTH_VISIBLE_GROUP = "0"
TERRAIN_RGBA = "0.72 0.52 0.30 1"
OBSTACLE_RGBA = "0.8 0.3 0.3 1"
HEIGHT_FIELD_RGBA = "0.5 0.5 0.55 1"
TERRAIN_FRICTION = "0.6 0.005 0.0001"
DEFAULT_RNG_SEED = 20260718


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles to a MuJoCo quaternion."""
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rot_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def rot2d(x: float, y: float, yaw: float) -> tuple[float, float]:
    return x * np.cos(yaw) - y * np.sin(yaw), x * np.sin(yaw) + y * np.cos(yaw)


def rot3d(pos: list[float], euler: list[float]) -> np.ndarray:
    return euler_to_rot(euler[0], euler[1], euler[2]) @ pos


def list_to_str(values: np.ndarray | list[float]) -> str:
    return " ".join(str(value) for value in values)


class TerrainGenerator:
    """Append collision-enabled, depth-visible terrain to a scene template."""

    def __init__(
        self,
        input_scene_path: Path | str = INPUT_SCENE_PATH,
        output_scene_path: Path | str = OUTPUT_SCENE_PATH,
        rng_seed: int = DEFAULT_RNG_SEED,
    ) -> None:
        self.input_scene_path = Path(input_scene_path)
        self.output_scene_path = Path(output_scene_path)
        self.scene = xml_et.parse(self.input_scene_path)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        if self.worldbody is None or self.asset is None:
            raise ValueError(
                f"{self.input_scene_path} must contain both <asset> and <worldbody> anchors"
            )
        self.rng = np.random.default_rng(rng_seed)

    def _stamp(self, geo: xml_et.Element, rgba: str = TERRAIN_RGBA) -> xml_et.Element:
        """Make generated terrain visible to the configured depth renderer."""
        geo.attrib["group"] = DEPTH_VISIBLE_GROUP
        geo.attrib["rgba"] = rgba
        geo.attrib.setdefault("contype", "1")
        geo.attrib.setdefault("conaffinity", "1")
        geo.attrib.setdefault("condim", "3")
        geo.attrib.setdefault("friction", TERRAIN_FRICTION)
        return geo

    def AddBox(
        self,
        position: list[float] = [1.0, 0.0, 0.0],
        euler: list[float] = [0.0, 0.0, 0.0],
        size: list[float] = [0.1, 0.1, 0.1],
    ) -> None:
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        geo.attrib["quat"] = list_to_str(euler_to_quat(*euler))
        self._stamp(geo)

    def AddGeometry(
        self,
        position: list[float] = [1.0, 0.0, 0.0],
        euler: list[float] = [0.0, 0.0, 0.0],
        size: list[float] = [0.1, 0.1],
        geo_type: str = "box",
        rgba: str = OBSTACLE_RGBA,
    ) -> None:
        """Add a Unitree-style geometry using full extents in ``size``."""
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        geo.attrib["quat"] = list_to_str(euler_to_quat(*euler))
        self._stamp(geo, rgba)

    def AddStairs(
        self,
        init_pos: list[float] = [1.0, 0.0, 0.0],
        yaw: float = 0.0,
        width: float = 0.2,
        height: float = 0.15,
        length: float = 1.5,
        stair_nums: int = 10,
    ) -> None:
        local_pos = [0.0, 0.0, -0.5 * height]
        for _ in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2]],
                [0.0, 0.0, yaw],
                [width, length, height],
            )

    def AddSuspendStairs(
        self,
        init_pos: list[float] = [1.0, 0.0, 0.0],
        yaw: float = 1.0,
        width: float = 0.2,
        height: float = 0.15,
        length: float = 1.5,
        gap: float = 0.1,
        stair_nums: int = 10,
    ) -> None:
        local_pos = [0.0, 0.0, -0.5 * height]
        for _ in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox(
                [x + init_pos[0], y + init_pos[1], local_pos[2]],
                [0.0, 0.0, yaw],
                [width, length, abs(height - gap)],
            )

    def AddRoughGround(
        self,
        init_pos: list[float] = [1.0, 0.0, 0.0],
        euler: list[float] = [0.0, 0.0, 0.0],
        nums: list[int] = [10, 10],
        box_size: list[float] = [0.5, 0.5, 0.5],
        box_euler: list[float] = [0.0, 0.0, 0.0],
        separation: list[float] = [0.2, 0.2],
        box_size_rand: list[float] = [0.05, 0.05, 0.05],
        box_euler_rand: list[float] = [0.2, 0.2, 0.2],
        separation_rand: list[float] = [0.05, 0.05],
    ) -> None:
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(separation_rand) * self.rng.uniform(-1.0, 1.0, 2)
        for _ in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for _ in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(box_size_rand) * self.rng.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(box_euler_rand) * self.rng.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(separation_rand) * self.rng.uniform(-1.0, 1.0, 2)
                local_pos[1] += new_separation[1]
                position = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(position.tolist(), new_box_euler.tolist(), new_box_size.tolist())

    @staticmethod
    def _height_field_dependencies():
        try:
            import cv2
            import noise
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Height-field generation requires optional dependencies: "
                "pip install noise opencv-python"
            ) from exc
        return cv2, noise

    def _add_hfield(
        self,
        name: str,
        position: list[float],
        euler: list[float],
        size: list[float],
        height_scale: float,
        negative_height: float,
        output_hfield_image: str,
    ) -> None:
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = name
        hfield.attrib["size"] = list_to_str([size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = name
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["quat"] = list_to_str(euler_to_quat(*euler))
        self._stamp(geo, HEIGHT_FIELD_RGBA)

    def AddPerlinHeighField(
        self,
        position: list[float] = [1.0, 0.0, 0.0],
        euler: list[float] = [0.0, 0.0, 0.0],
        size: list[float] = [1.0, 1.0],
        height_scale: float = 0.2,
        negative_height: float = 0.2,
        image_width: int = 128,
        img_height: int = 128,
        smooth: float = 100.0,
        perlin_octaves: int = 6,
        perlin_persistence: float = 0.5,
        perlin_lacunarity: float = 2.0,
        output_hfield_image: str = "height_field.png",
    ) -> None:
        cv2, noise = self._height_field_dependencies()
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(img_height):
            for x in range(image_width):
                value = noise.pnoise2(
                    x / smooth,
                    y / smooth,
                    octaves=perlin_octaves,
                    persistence=perlin_persistence,
                    lacunarity=perlin_lacunarity,
                )
                terrain_image[y, x] = int((value + 1) / 2 * 255)
        image_path = self.output_scene_path.parent / output_hfield_image
        if not cv2.imwrite(str(image_path), terrain_image):
            raise RuntimeError(f"Failed to write height-field image: {image_path}")
        self._add_hfield("perlin_hfield", position, euler, size, height_scale, negative_height, output_hfield_image)

    def AddHeighFieldFromImage(
        self,
        position: list[float] = [1.0, 0.0, 0.0],
        euler: list[float] = [0.0, 0.0, 0.0],
        size: list[float] = [2.0, 1.6],
        height_scale: float = 0.02,
        negative_height: float = 0.1,
        input_img: str | None = None,
        output_hfield_image: str = "height_field.png",
        image_scale: list[float] = [1.0, 1.0],
        invert_gray: bool = False,
    ) -> None:
        if input_img is None:
            raise ValueError("input_img is required for AddHeighFieldFromImage")
        cv2, _ = self._height_field_dependencies()
        input_image = cv2.imread(input_img)
        if input_image is None:
            raise FileNotFoundError(f"Unable to read height-field image: {input_img}")
        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - terrain_image
        image_path = self.output_scene_path.parent / output_hfield_image
        if not cv2.imwrite(str(image_path), terrain_image):
            raise RuntimeError(f"Failed to write height-field image: {image_path}")
        self._add_hfield("image_hfield", position, euler, size, height_scale, negative_height, output_hfield_image)

    def Save(self) -> None:
        self.output_scene_path.parent.mkdir(parents=True, exist_ok=True)
        xml_et.indent(self.scene, space="  ")
        self.scene.write(self.output_scene_path, encoding="utf-8", xml_declaration=True)


def add_default_course(generator: TerrainGenerator) -> None:
    """Build the deterministic, easy-to-hard course in the D435 +x view."""
    generator.AddStairs(
        init_pos=[1.2, 0.0, 0.0], width=0.20, height=0.05, length=1.20, stair_nums=4
    )
    generator.AddBox(
        position=[3.2, 0.0, 0.10], euler=[0.0, -0.12, 0.0], size=[1.20, 1.50, 0.04]
    )
    generator.AddRoughGround(
        init_pos=[5.0, -0.84, 0.0],
        nums=[7, 6],
        box_size=[0.25, 0.25, 0.10],
        box_size_rand=[0.04, 0.04, 0.04],
        box_euler_rand=[0.04, 0.04, 0.03],
        separation=[0.28, 0.28],
        separation_rand=[0.01, 0.01],
    )
    generator.AddGeometry(
        position=[7.6, 0.30, 0.40],
        euler=[0.0, 0.0, 0.0],
        size=[0.24, 0.80],
        geo_type="cylinder",
    )


if __name__ == "__main__":
    terrain_generator = TerrainGenerator()
    add_default_course(terrain_generator)
    terrain_generator.Save()
    print(f"wrote {OUTPUT_SCENE_PATH}")
