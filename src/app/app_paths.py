import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ROI_DIR = os.path.join(DATA_DIR, "roi")

ROI_PATH = os.path.join(ROI_DIR, "roi.json")
RECIPE_PATH = os.path.join(ROI_DIR, "recipe_static.json")
RUNTIME_CONFIG_PATH = os.path.join(ROI_DIR, "runtime_config.json")

LOGS_ROOT = os.path.join(DATA_DIR, "logs")