#### What this tests ####
#    This tests the validate environment function

import sys, os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm

print(litellm.validate_environment("openai/gpt-3.5-turbo"))
