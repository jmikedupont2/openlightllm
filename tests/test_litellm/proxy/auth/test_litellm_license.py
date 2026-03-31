import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

# REMOVED: from litellm.proxy.auth.litellm_license import LicenseCheck


def test_is_over_limit():


