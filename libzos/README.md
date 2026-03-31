# libzos-py — Python Plugin Interface for ZOS

Open source code imports from `libzos`. Enterprise features register as plugins.
No direct imports across the license boundary.

## Usage

```python
from libzos import plugin_registry, Plugin

# Use a plugin (open source side)
hasher = plugin_registry.get("monster_hash")
result = hasher.execute(b"hello")

# Register a plugin (enterprise/community side)
@plugin_registry.register("my_feature")
class MyFeature(Plugin):
    license = "Enterprise"
    def execute(self, data): ...
```

## Built-in Plugins

- `monster_hash`: 10-basin Monster barrel-shift hash
- `orbifold`: (mod 71, mod 59, mod 47) projection
- `license_check`: Cambridge theorem verification

## License

MIT
