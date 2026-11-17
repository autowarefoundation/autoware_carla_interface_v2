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

"""A copy of an Autoware map of a CARLA level that a GNSS receiver can localize in.

The CARLA maps' ``map_projector_info.yaml`` is ``Local``: Autoware reads the
lanelets' ``local_x``/``local_y`` and cannot place a latitude/longitude, so a
GNSS driver's NavSatFix means nothing to ``gnss_poser``. CARLA dates the ego's
geolocation with the level's OpenDRIVE geo-reference, which for the stock
levels is ``+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84``:
the map frame (CARLA's x and -y) is that projection's easting and northing.

This writes a map directory with

- ``map_projector_info.yaml``: ``TransverseMercator`` at origin (0, 0) with
  scale factor 1, the same projection;
- ``lanelet2_map.osm``: the source's, with each node's ``lat``/``lon``
  recomputed from its ``local_x``/``local_y`` (Autoware reads ``lat``/``lon``
  for a projector other than ``Local``; the source's are a little off), and
  without the bounds' duplicate points, which crash Autoware's intersection
  module (:func:`drop_duplicate_points`);
- the point cloud map and anything else linked from the source.

    hils_geo_map SOURCE_MAP_DIR OUTPUT_MAP_DIR
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from typing import Tuple

#: The stock CARLA levels' OpenDRIVE geo-reference.
CARLA_TMERC = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84"

PROJECTOR_INFO = """\
# The level's OpenDRIVE geo-reference ({proj}),
# which CARLA's geolocation and so a simulated GNSS receiver use.
projector_type: TransverseMercator
vertical_datum: WGS84
map_origin:
  latitude: {latitude}
  longitude: {longitude}
  altitude: 0.0
scale_factor: {scale_factor}
"""


def _inverse(proj: str):
    from pyproj import Transformer

    transformer = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    def to_geo(x: float, y: float):
        lon, lat = transformer.transform(x, y)
        return lat, lon

    return to_geo


#: Way points closer than this to their neighbour are duplicates (m).
DUPLICATE_POINT_DISTANCE = 1e-3


def drop_duplicate_points(root: ET.Element) -> int:
    """Remove a way's points that repeat their neighbour's position; return how many.

    The CARLA maps have bounds whose end point is followed (or preceded) by an
    interior point at the very same place. Autoware resamples each lanelet's
    bounds into its centerline, dividing by every segment's length: a zero one
    makes the centerline NaN, and the intersection module's search for the
    lanes before a conflicting lanelet then never ends (behavior_planning_container
    dies of a stack overflow). The end point is kept: neighbouring lanelets share it.
    """
    position = {}
    for node in root.iter("node"):
        tags = {tag.get("k"): tag.get("v") for tag in node.iter("tag")}
        if "local_x" in tags and "local_y" in tags:
            position[node.get("id")] = (
                float(tags["local_x"]),
                float(tags["local_y"]),
                float(tags.get("ele", tags.get("local_z", 0.0))),
            )
    dropped = 0
    for way in root.iter("way"):
        refs = way.findall("nd")
        keep = list(refs)
        i = 0
        while i + 1 < len(keep):
            a, b = position.get(keep[i].get("ref")), position.get(keep[i + 1].get("ref"))
            if a is None or b is None or math.dist(a, b) >= DUPLICATE_POINT_DISTANCE or len(keep) <= 2:
                i += 1
                continue
            # Drop the interior one of the pair; the end points stay.
            drop = i + 1 if i + 1 < len(keep) - 1 else i
            way.remove(keep.pop(drop))
            dropped += 1
    return dropped


def georeference_lanelets(source: str, output: str, proj: str = CARLA_TMERC) -> Tuple[int, int]:
    """Write ``source`` (a lanelet2 .osm) to ``output`` with lat/lon from local_x/local_y,
    and without its ways' duplicate points (:func:`drop_duplicate_points`).

    Returns the number of nodes rewritten and of duplicate points dropped.
    """
    to_geo = _inverse(proj)
    tree = ET.parse(source)
    count = 0
    for node in tree.getroot().iter("node"):
        tags = {tag.get("k"): tag.get("v") for tag in node.iter("tag")}
        if "local_x" not in tags or "local_y" not in tags:
            continue
        lat, lon = to_geo(float(tags["local_x"]), float(tags["local_y"]))
        node.set("lat", repr(lat))
        node.set("lon", repr(lon))
        count += 1
    dropped = drop_duplicate_points(tree.getroot())
    tree.write(output, encoding="UTF-8", xml_declaration=True)
    return count, dropped


def write_geo_map(source_dir: str, output_dir: str, proj: str = CARLA_TMERC) -> Tuple[int, int]:
    source_dir = os.path.abspath(source_dir)
    os.makedirs(output_dir, exist_ok=True)
    counts = georeference_lanelets(
        os.path.join(source_dir, "lanelet2_map.osm"), os.path.join(output_dir, "lanelet2_map.osm"), proj
    )
    with open(os.path.join(output_dir, "map_projector_info.yaml"), "w") as f:
        f.write(PROJECTOR_INFO.format(proj=proj, latitude=0.0, longitude=0.0, scale_factor=1.0))
    for name in sorted(os.listdir(source_dir)):
        if name in ("lanelet2_map.osm", "map_projector_info.yaml"):
            continue
        link = os.path.join(output_dir, name)
        if os.path.lexists(link):
            os.remove(link)
        # Relative, so the map also works where the data directory is mounted elsewhere (a container).
        os.symlink(os.path.relpath(os.path.join(source_dir, name), os.path.abspath(output_dir)), link)
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="Autoware map directory of the CARLA level (Local projector)")
    parser.add_argument("output", help="Directory to write the georeferenced map to")
    parser.add_argument("--proj", default=CARLA_TMERC, help="The level's OpenDRIVE geo-reference (only tmerc at 0/0)")
    args = parser.parse_args(argv)
    if "+proj=tmerc" not in args.proj or "+lat_0=0 " not in args.proj + " " or "+lon_0=0 " not in args.proj + " ":
        parser.error("Autoware's TransverseMercator projector here is at origin (0, 0): only such a tmerc is supported")
    count, dropped = write_geo_map(args.source, args.output, args.proj)
    print(f"{args.output}: {count} lanelet2 nodes georeferenced ({args.proj}), {dropped} duplicate points dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
