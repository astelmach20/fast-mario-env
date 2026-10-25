# Third-party notices

fast-mario-env is licensed under the MIT License (see `LICENSE`). It contains no code copied from the projects
below and ships no ROM; it re-implements their behaviour so that its output is identical to theirs, and its tests
use them as the reference. Their notices are reproduced here for attribution.

## nes-py

The NES emulator in `csrc/nes_core.hpp` (and its CUDA port in `src/fast_mario_env/cuda/kernels/`) is a
behaviour-identical re-implementation of the emulator in nes-py 8.2.1 (https://github.com/Kautenja/nes-py),
including its timing model and quirks. The NES master palette values are those used by nes-py.

```
MIT License

Copyright (c) 2019 Christian Kauten

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

nes-py's emulator is in turn derived from SimpleNES by Amish Naidu and contributors
(https://github.com/amhndu/SimpleNES), which is published under the GNU GPL v3. nes-py's own licensing notes
point out that this upstream declaration differs from nes-py's MIT declaration.

## gym-super-mario-bros

The reward, info, episode-end and level-selection semantics (and the RAM addresses they read) follow
gym-super-mario-bros 7.4.0 (https://github.com/Kautenja/gym-super-mario-bros). At runtime fast-mario-env can read
the ROM files that an installed gym-super-mario-bros package provides; it does not redistribute them.

```
Copyright (c) 2018 Christian Kauten

This project is provided for educational purposes only. It is not affiliated
with nor approved of by Nintendo Co., Ltd.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenCV and NumPy

The grayscale conversion and `INTER_AREA` resize reproduce OpenCV's (https://opencv.org, Apache License 2.0)
results by following its published algorithm and operation order, and the sticky-action / no-op random numbers
reproduce NumPy's PCG64 `Generator` (https://numpy.org, BSD 3-Clause). No code from either project is included.

## Nintendo

Super Mario Bros. is a trademark of Nintendo. This project is not affiliated with, endorsed by or approved by
Nintendo. No game ROM is included in this repository or its packages.
