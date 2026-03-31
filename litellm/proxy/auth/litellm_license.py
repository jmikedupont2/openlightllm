# What is this?
## If litellm license in env, checks if it's valid
import base64
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.constants import NON_LLM_CONNECTION_TIMEOUT
from litellm.llms.custom_httpx.http_handler import HTTPHandler

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData

