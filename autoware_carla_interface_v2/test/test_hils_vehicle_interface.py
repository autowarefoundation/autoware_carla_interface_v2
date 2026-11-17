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

"""The HILS vehicle interface's conversions between the CAN and Autoware."""

import math
import os

import pytest

from autoware_carla_interface_v2.hils_vehicle_interface import MKZ_MAX_STEER_ANGLE
from autoware_carla_interface_v2.hils_vehicle_interface import Settle
from autoware_carla_interface_v2.hils_vehicle_interface import control_cmd_fields
from autoware_carla_interface_v2.hils_vehicle_interface import tire_angle_from_steer
from autoware_carla_interface_v2.vehicle_interface import GEAR_DRIVE
from autoware_carla_interface_v2.vehicle_interface import GEAR_PARK
from autoware_carla_interface_v2.vehicle_interface import GEAR_REVERSE
from autoware_carla_interface_v2.vehicle_interface import reported_gear

DBC = os.path.join(os.path.dirname(__file__), "..", "config", "hils", "carla_autoware.dbc")
TAMAGAWA_DBC = os.path.join(os.path.dirname(__file__), "..", "config", "hils", "tamagawa_imu.dbc")


def test_tire_angle_is_left_positive_and_scaled_to_the_wheel_limit():
    assert tire_angle_from_steer(1.0) == pytest.approx(-MKZ_MAX_STEER_ANGLE)
    assert tire_angle_from_steer(-0.5, 1.0) == pytest.approx(0.5)


def test_control_cmd_keeps_the_tire_angle_left_positive_on_the_bus():
    fields = control_cmd_fields(0.2, 3.0, -1.5, 0.1, GEAR_DRIVE)
    assert fields == pytest.approx({"TargetSpeed": 3.0, "Accel": 1.5, "SteerAngle": 0.2, "Jerk": 0.1})


def test_control_cmd_reverses_the_target_speed_in_reverse():
    assert control_cmd_fields(0.0, 2.0, 0.5, 0.0, GEAR_REVERSE)["TargetSpeed"] == pytest.approx(-2.0)


def test_reported_gear_follows_reverse_and_park():
    assert reported_gear(GEAR_DRIVE, False) == GEAR_DRIVE
    assert reported_gear(GEAR_DRIVE, True) == GEAR_REVERSE
    assert reported_gear(GEAR_PARK, False) == GEAR_PARK


def test_the_dbc_round_trips_the_messages_the_bridge_and_the_node_exchange():
    hilspy = pytest.importorskip("hilspy")
    with hilspy.open("mem://hils-test", f"dbc:{DBC}") as tx, hilspy.open("mem://hils-test", f"dbc:{DBC}") as rx:
        tx.send("ControlCmd", control_cmd_fields(-0.1234, 4.2, 0.5, 0.0, GEAR_DRIVE))
        control = rx.recv(timeout=1.0)
    assert control.name == "ControlCmd"
    assert control.fields["SteerAngle"] == pytest.approx(-0.1234, abs=1e-4)
    assert control.fields["TargetSpeed"] == pytest.approx(4.2, abs=0.01)


def test_vehicle_status_carries_the_yaw_rate():
    hilspy = pytest.importorskip("hilspy")
    with hilspy.open("mem://hils-status", f"dbc:{DBC}") as tx, hilspy.open("mem://hils-status", f"dbc:{DBC}") as rx:
        tx.send("VehicleStatus", {"Speed": 3.5, "Steer": 0.1, "Gear": 1, "Reverse": 0, "HandBrake": 0, "YawRate": -0.25})
        status = rx.recv(timeout=1.0)
    assert status.name == "VehicleStatus"
    assert status.fields["YawRate"] == pytest.approx(-0.25, abs=1e-4)


def _tag_can_driver_decode(gyro: bytes, accel: bytes):
    """What tamagawa_imu_driver's tag_can_driver makes of 0x319 and 0x31A."""

    def i16(data, k):
        return int.from_bytes(data[k:k + 2], "big", signed=True)

    rates = [i16(gyro, k) * (200 / 2**15) * math.pi / 180 for k in (2, 4, 6)]
    acceleration = [i16(accel, k) * (100 / 2**15) for k in (2, 4, 6)]
    return int.from_bytes(gyro[0:2], "big"), rates, acceleration


def test_the_tamagawa_dbc_encodes_what_tag_can_driver_decodes():
    hilspy = pytest.importorskip("hilspy")
    with hilspy.open("mem://hils-imu", f"dbc:{TAMAGAWA_DBC}") as tx, hilspy.open("mem://hils-imu") as rx:
        tx.send("TAG_Gyro", {"Counter": 513, "GyroX": 12.5, "GyroY": -3.25, "GyroZ": 150.0})
        tx.send("TAG_Accel", {"Counter": 514, "AccelX": -1.5, "AccelY": 0.75, "AccelZ": -9.80665})
        gyro = rx.recv(timeout=1.0)
        accel = rx.recv(timeout=1.0)
    assert (gyro.id, accel.id) == (0x319, 0x31A)
    counter, rates, acceleration = _tag_can_driver_decode(gyro.data, accel.data)
    assert counter == 513
    lsb_rate = math.radians(200 / 2**15)
    assert rates == pytest.approx([math.radians(12.5), math.radians(-3.25), math.radians(150.0)], abs=lsb_rate)
    assert acceleration == pytest.approx([-1.5, 0.75, -9.80665], abs=100 / 2**15)


def test_settle_needs_the_condition_to_hold_unbroken():
    settle = Settle(2.0)
    assert not settle.update(True, 10.0)
    assert not settle.update(True, 11.9)
    assert settle.update(True, 12.0)
    # A single break (the mode flickering unavailable) starts the wait over.
    assert not settle.update(False, 12.1)
    assert not settle.update(True, 12.2)
    assert not settle.update(True, 14.1)
    assert settle.update(True, 14.2)
