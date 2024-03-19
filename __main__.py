import os
import json

settings = {}
with open("./config.json","r") as file:
    settings = json.loads(file)

# 1. Organ masking
# 2. Image cropping
# 3. Patch normalization
# 4. Segmentation rebuilding
# 5. Segmentation analysis
# 6. Whole body heatmap generation
