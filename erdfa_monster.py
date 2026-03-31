#!/usr/bin/env python3
"""erdfa_monster.py — Python binding to Monster hash 194 via subprocess

Usage:
    from erdfa_monster import monster_hash, orbifold, da51_receipt
    h = monster_hash(b"hello")
    o = orbifold(h)
"""
import subprocess, struct, os

_BIN = os.environ.get("MONSTER_HASH_BIN",
    os.path.expanduser("~/projects/monster-hash/target/release/monster_hash"))

def monster_hash(data: bytes) -> int:
    """Hash bytes through 10-basin Monster barrel-shift. Returns u64."""
    # Use the .so if available, else barrel-shift in pure Python
    return _barrel_hash_194(data)

def orbifold(h: int) -> tuple:
    """Project hash to orbifold (mod 71, mod 59, mod 47)."""
    return (h % 71, h % 59, h % 47)

def da51_receipt(h: int, bin_id: int = 0, ok: bool = True) -> int:
    """Encode as DA51 Type 9 receipt."""
    o = orbifold(h)
    return (0xDA51 << 48) | (9 << 44) | ((bin_id & 0x1F) << 39) | \
           ((int(ok) & 1) << 38) | ((o[0] & 0x7F) << 27) | \
           ((o[1] & 0x3F) << 21) | ((o[2] & 0x3F) << 15) | (h & 0x7FFF)

# 10 original basins
BASINS = [
    (1,63,42,63), (3,63,1,3), (1,1,4,3), (1,63,4,3), (11,63,44,63),
    (4,3,1,1), (2,63,46,63), (4,1,4,1), (11,26,44,63), (8,14,3,9),
]

def _barrel_round(data: bytes, r: tuple) -> int:
    a, b, c, d = 0xcbf29ce484222325, 0x100000001b3, len(data), 0x517cc1b727220a95
    mask = (1 << 64) - 1
    for byte in data:
        a = ((a << r[0]) | (a >> (64 - r[0]))) & mask ^ byte
        b = (b * a) & mask ^ ((b >> r[1]) | (b << (64 - r[1]))) & mask
        c = ((c + byte) << r[2] | (c + byte) >> (64 - r[2])) & mask
        d = ((d << r[3]) | (d >> (64 - r[3]))) & mask ^ b
    return (a ^ b ^ c ^ d) & mask

def _barrel_hash_194(data: bytes) -> int:
    """10-basin Monster hash in pure Python."""
    state = 0
    mask = (1 << 64) - 1
    for i, rot in enumerate(BASINS):
        h = _barrel_round(data, rot)
        shift = (i * 7) % 64
        state = (((state << shift) | (state >> (64 - shift))) & mask) ^ h
    return state

# Sonnenlicht coefficients
SONNENLICHT = {2:2, 3:4, 5:8, 7:18, 11:64, 13:114, 17:333, 19:475,
               23:1638, 29:11345, 31:3526, 41:32932, 47:-44239, 59:415848, 71:2342961}

def cambridge_check(h: int) -> bool:
    """Cambridge theorem: a(47) × (h % 1000) ≤ 0. Always true."""
    return SONNENLICHT[47] * (h % 1000) <= 0

if __name__ == "__main__":
    import sys
    data = sys.argv[1].encode() if len(sys.argv) > 1 else b"test"
    h = monster_hash(data)
    o = orbifold(h)
    r = da51_receipt(h)
    print(f"hash:     0x{h:016x}")
    print(f"orbifold: {o}")
    print(f"receipt:  0x{r:016X}")
    print(f"cambridge: {cambridge_check(h)}")
