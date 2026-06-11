#!/usr/bin/env python3
"""Read room-ambient temperature from the bench rig's PCsensor TEMPer2 (3553:a001,
firmware TEMPer2_V4.1) and print ONE number: degrees C.

Plugs into gpu_bench.py via --ambient-cmd; no dependencies (raw hidraw).

The stick has two sensors: internal (in the USB body — self-heats a couple of
degrees next to the chassis) and the external probe on the wire. Room ambient is
the EXTERNAL probe (keep it hanging in free air, away from the rig/dock exhaust);
internal is only a fallback if the probe is unplugged. Layout matches the
TEMPer2_V3.7/V3.9 parse in temper-py (which doesn't know V4.1): two 8-byte HID
reports, internal temp at bytes 2-3, external at bytes 10-11, both /100;
0x4e20 (200.00) is the no-sensor sentinel.
"""
import os
import select
import struct
import sys

HID_ID = "00003553:0000A001"  # zero-padded VID:PID as it appears in hidraw uevent
QUERY = b"\x01\x80\x33\x01\x00\x00\x00\x00"
SENTINEL = 0x4E20  # 200.00 C = sensor absent


def find_hidraw():
    """Last hidraw node of the TEMPer2 (the data interface, per temper-py)."""
    nodes = []
    base = "/sys/class/hidraw"
    for name in sorted(os.listdir(base)):
        try:
            with open(f"{base}/{name}/device/uevent") as f:
                uevent = f.read()
        except OSError:
            continue
        if HID_ID in uevent.upper():
            nodes.append(f"/dev/{name}")
    if not nodes:
        raise SystemExit("TEMPer2 (3553:a001) not found on any hidraw node")
    return nodes[-1]


def read_reports(dev):
    fd = os.open(dev, os.O_RDWR)
    try:
        os.write(fd, QUERY)
        data = b""
        while len(data) < 16:
            r, _, _ = select.select([fd], [], [], 2.0)
            if not r:
                break
            data += os.read(fd, 8)
        return data
    finally:
        os.close(fd)


def main():
    data = read_reports(find_hidraw())
    if len(data) < 8:
        raise SystemExit(f"short read from TEMPer2: {data.hex()}")
    internal = struct.unpack_from(">h", data, 2)[0]
    external = struct.unpack_from(">h", data, 10)[0] if len(data) >= 12 else SENTINEL
    for raw, label in ((external, "external"), (internal, "internal")):
        if raw != SENTINEL:
            if "--verbose" in sys.argv:
                print(f"{raw / 100.0:.2f} ({label}, raw {data.hex()})")
            else:
                print(f"{raw / 100.0:.2f}")
            return
    raise SystemExit(f"no live sensor (both sentinel): {data.hex()}")


if __name__ == "__main__":
    main()
