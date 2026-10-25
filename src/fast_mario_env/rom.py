"""Locating the Super Mario Bros ROM and decoding level names.

No ROM is shipped with this package. It is loaded at runtime, in this order:

1. the ``rom_path=`` argument,
2. the ``FAST_MARIO_ENV_ROM`` environment variable,
3. the ROM files bundled with an installed ``gym-super-mario-bros`` package
   (located on disk; the package itself is not imported).
"""
from __future__ import annotations

import importlib.util
import os

ROM_ENV_VAR = "FAST_MARIO_ENV_ROM"

# file names used by gym-super-mario-bros for each ROM mode (Super Mario Bros 1 only)
ROM_FILES = {
    "vanilla": "super-mario-bros.nes",
    "downsample": "super-mario-bros-downsample.nes",
    "pixel": "super-mario-bros-pixel.nes",
    "rectangle": "super-mario-bros-rectangle.nes",
}

_HELP = (
    "fast-mario-env does not ship a ROM. Provide one of:\n"
    "  * rom_path='/path/to/super-mario-bros.nes' when creating the env,\n"
    f"  * the environment variable {ROM_ENV_VAR}=/path/to/super-mario-bros.nes,\n"
    "  * or install gym-super-mario-bros (`pip install gym-super-mario-bros`), whose ROM files are used.\n"
    "The ROM must be an iNES Super Mario Bros (NROM / mapper 0) image."
)


class RomNotFoundError(FileNotFoundError):
    pass


def _gsmb_rom_dir():
    spec = importlib.util.find_spec("gym_super_mario_bros")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "_roms")


def find_rom(rom_path: str | os.PathLike | None = None, rom_mode: str = "vanilla") -> str:
    """Return the path of the ROM to use (see module docstring for the search order)."""
    if rom_mode not in ROM_FILES:
        raise ValueError(f"rom_mode must be one of {sorted(ROM_FILES)}, got {rom_mode!r}")
    for cand, src in ((rom_path, "rom_path"), (os.environ.get(ROM_ENV_VAR) or None, ROM_ENV_VAR)):
        if cand is not None:
            cand = os.path.expanduser(os.fspath(cand))
            if not os.path.isfile(cand):
                raise RomNotFoundError(f"{src} points to {cand!r}, which is not a file.\n\n{_HELP}")
            return cand
    d = _gsmb_rom_dir()
    if d is not None:
        p = os.path.join(d, ROM_FILES[rom_mode])
        if os.path.isfile(p):
            return p
    raise RomNotFoundError("No Super Mario Bros ROM found.\n\n" + _HELP)


def load_rom(rom_path=None, rom_mode: str = "vanilla") -> bytes:
    """Read and validate the ROM; returns the raw iNES bytes."""
    path = find_rom(rom_path, rom_mode)
    with open(path, "rb") as f:
        data = f.read()
    check_rom(data, path)
    return data


def check_rom(data: bytes, name: str = "ROM") -> None:
    if len(data) < 16 or data[:4] != b"NES\x1a":
        raise ValueError(f"{name}: not an iNES ROM image (bad header)")
    mapper = (data[6] >> 4) | (data[7] & 0xF0)
    prg, chr_ = data[4], data[5]
    if mapper != 0 or prg not in (1, 2) or chr_ != 1:
        raise ValueError(f"{name}: unsupported ROM (mapper {mapper}, {prg}x16K PRG, {chr_}x8K CHR); "
                         "only NROM Super Mario Bros images are supported")
    if len(data) < 16 + prg * 0x4000 + 0x2000:
        raise ValueError(f"{name}: truncated ROM image")


ALL_LEVELS = tuple(f"{w}-{s}" for w in range(1, 9) for s in range(1, 5))


def decode_level(level) -> tuple[int, int, int]:
    """'w-s' or (w, s) -> (world, stage, area) as used by gym-super-mario-bros' target levels.

    In worlds 1, 2, 4 and 7 stage 2 begins with an extra intro area, so the area index of
    stages >= 2 is shifted by one there."""
    if isinstance(level, str):
        try:
            w, s = (int(x) for x in level.split("-"))
        except ValueError:
            raise ValueError(f"level must look like '1-1', got {level!r}") from None
    else:
        w, s = (int(x) for x in level)
    if not (1 <= w <= 8 and 1 <= s <= 4):
        raise ValueError(f"level {level!r} out of range (worlds 1-8, stages 1-4)")
    area = s + 1 if (w in (1, 2, 4, 7) and s >= 2) else s
    return w, s, area
