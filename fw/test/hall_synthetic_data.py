#!/usr/bin/env python3

# Copyright 2026 mjbots Robotic Systems, LLC.  info@mjbots.com
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

"""Generate synthetic hall sensor captures for hall_regression_test.

The real captures in fw/test/data are all default-polarity motors
driven with mild accelerations.  These synthetic streams exercise the
regimes those cannot: an inverted hall line, where the late
(pull-up limited) rising edges no longer alternate with the sharp
falling ones, and hard accelerations that force repeated transitions
between the slow-mode and PLL velocity estimators.

The output is the same raw debug-port dump format the real captures
use: one 3-byte record per control cycle at 30 kHz, a 0x5a header
followed by the little-endian 16 bit raw hall pin state.  Each stream
models:

  * a piecewise-linear profile of hall edge rate versus time,
  * a fixed per-sector width pattern (one electrical revolution of the
    90 counts/rev test motor) drawn uniformly within +/- jitter_pct,
  * a rising-edge delay: every transition where a physical line
    rises is registered late by rise_delay_us,
  * the pin polarity: the physical bits are the logical 6-step
    sequence XOR polarity, and the delay applies to the physical
    rising edges, exactly as an inverted open-collector line behaves.

Running the script regenerates every dataset listed in DATASETS in
fw/test/data.  The generation is deterministic.
"""

import gzip
import math
import os
import struct

import numpy as np

RATE_HZ = 30000.0
COUNTS_PER_REV = 90
SECTOR_BITS = [0b001, 0b011, 0b010, 0b110, 0b100, 0b101]

DATASETS = {
    # Hall line A inverted.  Slow mode, a ramp, PLL mode, a ramp back.
    '20260914-synth-polarity-inverted-a': dict(
        polarity=0b001, rise_delay_us=120.0, jitter_pct=3.0, seed=1,
        profile=[(0.0, 200.0), (1.5, 200.0), (2.0, 1500.0),
                 (4.0, 1500.0), (4.5, 200.0), (5.5, 200.0)]),
    # Trapezoids between 300 and 2000 edges/s with 0.1 s (and shorter)
    # ramps, i.e. 17000 edges/s^2 through the slow/PLL boundary.
    '20260914-synth-accel-trapezoid': dict(
        polarity=0, rise_delay_us=120.0, jitter_pct=3.0, seed=1,
        profile=[(0.0, 300.0), (0.5, 300.0), (0.6, 2000.0),
                 (1.1, 2000.0), (1.2, 300.0), (1.7, 300.0),
                 (1.75, 2000.0), (2.25, 2000.0), (2.3, 300.0),
                 (2.8, 300.0), (2.82, 2000.0), (3.3, 2000.0),
                 (3.32, 300.0), (3.8, 300.0)]),
    # A 3 Hz sinusoidal speed between 200 and 2000 edges/s.
    '20260914-synth-accel-sine': dict(
        polarity=0, rise_delay_us=120.0, jitter_pct=3.0, seed=1,
        profile=[(t, 1100.0 + 900.0 * math.sin(2.0 * math.pi * 3.0 * t))
                 for t in np.linspace(0.0, 4.0, 81)]),
}


def generate(polarity, rise_delay_us, jitter_pct, seed, profile):
    rng = np.random.default_rng(seed)
    widths = 1.0 + jitter_pct / 100.0 * rng.uniform(-1.0, 1.0, COUNTS_PER_REV)
    widths *= COUNTS_PER_REV / widths.sum()

    times = np.array([p[0] for p in profile])
    rates = np.array([p[1] for p in profile])
    n = int(times[-1] * RATE_HZ)
    t = np.arange(n) / RATE_HZ
    # Ideal edge count as a function of time.
    phase = np.cumsum(np.interp(t, times, rates)) / RATE_HZ

    # Ideal (on-time) edge times, each edge advancing by its sector's
    # width.
    boundaries = np.cumsum([widths[k % COUNTS_PER_REV]
                            for k in range(int(phase[-1]) + 1)])
    edge_times = np.searchsorted(phase, boundaries[boundaries < phase[-1]]) / RATE_HZ

    raw = np.full(n, SECTOR_BITS[0] ^ polarity, dtype=np.uint16)
    for k, te in enumerate(edge_times):
        old_bits = SECTOR_BITS[k % 6] ^ polarity
        new_bits = SECTOR_BITS[(k + 1) % 6] ^ polarity
        rising = (new_bits & (new_bits ^ old_bits)) != 0
        i = int(np.searchsorted(
            t, te + (rise_delay_us * 1e-6 if rising else 0.0)))
        if i >= n:
            break
        raw[i:] = new_bits
    return b''.join(struct.pack('<BH', 0x5a, int(r)) for r in raw)


def main():
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    for name, params in DATASETS.items():
        path = os.path.join(datadir, name + '.dat.gz')
        # mtime=0 keeps the output byte-identical across runs.
        with open(path, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb',
                               compresslevel=9, mtime=0) as f:
                f.write(generate(**params))
        print(path)


if __name__ == '__main__':
    main()
