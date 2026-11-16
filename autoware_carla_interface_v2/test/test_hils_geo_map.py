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

import pytest
import yaml

from autoware_carla_interface_v2.hils_geo_map import write_geo_map

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

    assert write_geo_map(str(source), str(output)) == 1

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
