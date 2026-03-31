"""zos_litellm_patch.py — Replace enterprise loader with ZOS plugin registry

Patches litellm to:
1. Replace enterprise imports with libzos plugin lookups
2. Add zkperf timing to every provider call
3. Emit DA51 orbifold coords in HTTP response headers
4. Publish perf data via erdfa-publish

Usage:
    import zos_litellm_patch  # apply before importing litellm
    import litellm
"""
import time, functools
from erdfa_monster import monster_hash, orbifold, da51_receipt

# ── 1. ZOS Plugin Registry (replaces enterprise loader) ──

from libzos import plugin_registry, Plugin

def zos_get_enterprise_feature(feature_name):
    """Drop-in replacement for litellm's enterprise feature loader.
    Returns the plugin if installed, None otherwise. No license crossing."""
    return plugin_registry.get(feature_name)

# ── 2. zkperf wrapper for provider calls ──

_perf_log = []

def zkperf_wrap(fn):
    """Wrap any provider call with timing + orbifold tracking."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            result = fn(*args, **kwargs)
            ok = True
        except Exception as e:
            ok = False
            raise
        finally:
            elapsed_ns = time.perf_counter_ns() - t0
            # Hash the call signature
            call_sig = f"{fn.__module__}.{fn.__qualname__}".encode()
            h = monster_hash(call_sig)
            o = orbifold(h)
            receipt = da51_receipt(h, ok=ok)

            entry = {
                "fn": fn.__qualname__,
                "ns": elapsed_ns,
                "hash": f"0x{h:016x}",
                "orbifold": o,
                "receipt": f"0x{receipt:016X}",
                "ok": ok,
            }
            _perf_log.append(entry)

        return result
    return wrapper

def get_perf_log():
    """Get accumulated perf entries."""
    return _perf_log

def clear_perf_log():
    """Clear perf log."""
    _perf_log.clear()

# ── 3. DA51 HTTP headers ──

def da51_headers(response_data: bytes = b"") -> dict:
    """Generate DA51 headers for HTTP response.
    Add these to every LLM proxy response."""
    h = monster_hash(response_data[:256] if response_data else b"empty")
    o = orbifold(h)
    receipt = da51_receipt(h)

    # Perf summary from recent calls
    recent = _perf_log[-10:] if _perf_log else []
    total_ns = sum(e["ns"] for e in recent)
    avg_ns = total_ns // max(len(recent), 1)

    return {
        "X-DA51-Hash": f"0x{h:016x}",
        "X-DA51-Orbifold": f"{o[0]},{o[1]},{o[2]}",
        "X-DA51-Receipt": f"0x{receipt:016X}",
        "X-DA51-Perf-Ns": str(avg_ns),
        "X-DA51-Calls": str(len(recent)),
    }

# ── 4. erdfa-publish perf dataset ──

def publish_perf(spool_dir="/mnt/data1/spool/uucp/pastebin"):
    """Publish accumulated perf data as DA51 shards."""
    import json
    from pathlib import Path

    if not _perf_log:
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    lines = []
    for e in _perf_log:
        o = e["orbifold"]
        lines.append(f"Sheaf: {o[0]},{o[1]},{o[2]} H/raw p=1 ns={e['ns']} {e['fn']}")

    # Write corpus
    corpus = "\n".join(lines)
    h = monster_hash(corpus.encode())
    o = orbifold(h)

    shard = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "witness": f"0x{h:016x}",
        "orbifold": o,
        "calls": len(_perf_log),
        "total_ns": sum(e["ns"] for e in _perf_log),
        "entries": _perf_log[-100],  # last 100
    }

    path = Path(spool_dir) / f"{ts}_zos_litellm_perf.txt"
    path.write_text(corpus)

    return shard

# ── 5. Monkey-patch litellm (call after import) ──

def patch_litellm():
    """Apply ZOS patches to litellm. Call once at startup."""
    try:
        import litellm

        # Replace enterprise feature loader
        if hasattr(litellm, '_get_enterprise_feature'):
            litellm._get_enterprise_feature = zos_get_enterprise_feature

        # Wrap completion with zkperf
        if hasattr(litellm, 'completion'):
            litellm.completion = zkperf_wrap(litellm.completion)
        if hasattr(litellm, 'acompletion'):
            litellm.acompletion = zkperf_wrap(litellm.acompletion)
        if hasattr(litellm, 'embedding'):
            litellm.embedding = zkperf_wrap(litellm.embedding)

        print("[ZOS] litellm patched: enterprise→plugin, zkperf enabled, DA51 headers ready")
        return True
    except ImportError:
        print("[ZOS] litellm not installed, patch skipped")
        return False

if __name__ == "__main__":
    print("═══ ZOS LITELLM PATCH ═══\n")
    print("  Replaces: enterprise loader → libzos plugin registry")
    print("  Adds:     zkperf timing on every provider call")
    print("  Emits:    DA51 orbifold coords in HTTP headers")
    print("  Publishes: perf data to spool as DA51 shards\n")

    # Demo headers
    headers = da51_headers(b"test response from gpt-4")
    for k, v in headers.items():
        print(f"  {k}: {v}")

    print(f"\n  Plugins: {plugin_registry.list()}")
