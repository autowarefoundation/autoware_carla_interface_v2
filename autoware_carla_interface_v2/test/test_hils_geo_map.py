# Copyright 2026 Tier IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""hils_geo_map: a CARLA level's Autoware map that a GNSS receiver can localize in."""

import os
import xml.etree.ElementTree as ET

import pytest
import yaml

from autoware_carla_interface_v2.hils_geo_map import drop_duplicate_points, write_geo_map

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="8.4440784312e-05" lon="0.00097084469596" version="1" visible="True">
    <tag k="local_y" v="9.34613037109"/>
    <tag k="local_x" v="108.179916382"/>
    <tag k="ele" v="0.0"/>
  </node>
  <node id="2" lat="0.0" lon="0.0" version="1" visible="True"/>
</osm>
"""


def test_lanelet_nodes_get_carlas_geolocation_and_the_rest_is_linked(tmp_path):
    pytest.importorskip("pyproj")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "lanelet2_map.osm").write_text(OSM)
    (source / "map_projector_info.yaml").write_text("projector_type: Local\n")
    (source / "pointcloud_map.pcd").write_text("pcd")

    assert write_geo_map(str(source), str(output)) == (1, 0)

    info = yaml.safe_load((output / "map_projector_info.yaml").read_text())
    assert info["projector_type"] == "TransverseMercator"
    assert info["map_origin"]["latitude"] == 0.0 and info["map_origin"]["longitude"] == 0.0
    assert info["scale_factor"] == 1.0
    # CARLA's map.transform_to_geolocation(carla.Location(108.179916382, -9.34613037109, 0))
    # on Town10HD_Opt: map (x, y) is CARLA's (x, -y).
    text = (output / "lanelet2_map.osm").read_text()
    lat = float(text.split('lat="')[1].split('"')[0])
    lon = float(text.split('lon="')[1].split('"')[0])
    assert lat == pytest.approx(8.452355034742886e-05, abs=1e-12)
    assert lon == pytest.approx(0.0009717967231598924, abs=1e-12)
    # A node without local coordinates is left as it was.
    assert 'lat="0.0" lon="0.0"' in text
    link = output / "pointcloud_map.pcd"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert link.read_text() == "pcd"


def _node(node_id, x, y):
    return (
        f'<node id="{node_id}" lat="0" lon="0"><tag k="local_x" v="{x}"/>'
        f'<tag k="local_y" v="{y}"/><tag k="ele" v="0.0"/></node>'
    )


def _refs(way):
    return [nd.get("ref") for nd in way.findall("nd")]


def test_a_bound_loses_the_interior_point_that_repeats_its_end_point():
    # Town10HD_Opt way 4638 (the left bound of lanelet 4640, which crosses the
    # right turn 4766) starts with node 1173 and then node 4550 at the same
    # place: Autoware's centerline of it came out NaN.
    root = ET.fromstring(
        "<osm>"
        + _node(1, 0.0, 0.0) + _node(2, 0.0, 0.0) + _node(3, 5.0, 0.0)
        + _node(4, 10.0, 0.0) + _node(5, 10.0, 0.0)
        + '<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="5"/></way>'
        + '<way id="11"><nd ref="1"/><nd ref="3"/></way>'
        + "</osm>"
    )

    assert drop_duplicate_points(root) == 2

    ways = {way.get("id"): way for way in root.iter("way")}
    # The end points, which the neighbouring lanelets share, stay.
    assert _refs(ways["10"]) == ["1", "3", "5"]
    assert _refs(ways["11"]) == ["1", "3"]
