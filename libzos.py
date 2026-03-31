"""libzos — Python plugin interface for ZOS

Open source code imports from libzos. Enterprise features register as plugins.
No direct imports across the license boundary.

Usage:
    from libzos import plugin_registry, Plugin

    # Register a plugin (done by Enterprise or community)
    @plugin_registry.register("guardrails")
    class MyGuardrail(Plugin):
        def execute(self, data): ...

    # Use a plugin (done by open source code)
    guardrail = plugin_registry.get("guardrails")
    if guardrail:
        result = guardrail.execute(data)
"""
from erdfa_monster import monster_hash, orbifold, da51_receipt

class Plugin:
    """Base plugin interface. All plugins implement this."""
    name: str = "unnamed"
    license: str = "MIT"

    def execute(self, data: bytes) -> bytes:
        raise NotImplementedError

    def orbifold_id(self) -> tuple:
        """Each plugin has a unique orbifold coordinate."""
        return orbifold(monster_hash(self.name.encode()))

    def da51_addr(self) -> int:
        """DA51 address for this plugin."""
        h = monster_hash(self.name.encode())
        return da51_receipt(h, bin_id=5)  # Type 5 = Shard

class PluginRegistry:
    """Registry of available plugins. Decouples open source from enterprise."""

    def __init__(self):
        self._plugins: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a plugin class."""
        def decorator(cls):
            cls.name = name
            self._plugins[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Plugin | None:
        """Get a plugin instance by name. Returns None if not installed."""
        cls = self._plugins.get(name)
        return cls() if cls else None

    def list(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def orbifold_map(self) -> dict[str, tuple]:
        """Map of plugin name → orbifold coordinate."""
        result = {}
        for name, cls in self._plugins.items():
            inst = cls()
            result[name] = inst.orbifold_id()
        return result

# Global registry
plugin_registry = PluginRegistry()

# Built-in plugins (open source, always available)
@plugin_registry.register("monster_hash")
class MonsterHashPlugin(Plugin):
    license = "MIT"
    def execute(self, data: bytes) -> bytes:
        h = monster_hash(data)
        return h.to_bytes(8, "little")

@plugin_registry.register("orbifold")
class OrbifoldPlugin(Plugin):
    license = "MIT"
    def execute(self, data: bytes) -> bytes:
        h = monster_hash(data)
        o = orbifold(h)
        return f"{o[0]},{o[1]},{o[2]}".encode()

@plugin_registry.register("license_check")
class LicenseCheckPlugin(Plugin):
    license = "MIT"
    def execute(self, data: bytes) -> bytes:
        from erdfa_monster import SONNENLICHT
        h = monster_hash(data[:512])
        q47 = SONNENLICHT[47] * (h % 1000)
        return b"cambridge_ok" if q47 <= 0 else b"cambridge_fail"

if __name__ == "__main__":
    print("═══ LIBZOS PLUGIN REGISTRY ═══\n")
    for name in plugin_registry.list():
        p = plugin_registry.get(name)
        o = p.orbifold_id()
        print(f"  {name:>20}: ({o[0]:>2},{o[1]:>2},{o[2]:>2}) license={p.license}")
    print(f"\n  Total: {len(plugin_registry.list())} plugins")
    print(f"\n  Enterprise plugins register at runtime via:")
    print(f"    @plugin_registry.register('my_feature')")
    print(f"    class MyFeature(Plugin): ...")
