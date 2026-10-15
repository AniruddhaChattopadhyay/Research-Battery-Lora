"""Print the total OpenRouter usage in dollars (to measure the cost of each run)."""
import json
import os
import sys
import urllib.request

import env  # noqa: F401  (loads the API keys from .env)

req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                             headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
usage = json.load(urllib.request.urlopen(req))["data"]["total_usage"]
print("%.4f" % usage)
if len(sys.argv) > 1:
    with open(sys.argv[1], "a") as f:
        f.write(json.dumps({"label": sys.argv[2], "usage": usage}) + "\n")
