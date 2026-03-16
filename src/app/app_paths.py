import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ROI_DIR = os.path.join(DATA_DIR, "roi")

ROI_PATH = os.path.join(ROI_DIR, "roi.json")
RECIPE_PATH = os.path.join(ROI_DIR, "recipe_static.json")
RECIPES_DIR = os.path.join(ROI_DIR, "recipes")
DEFAULT_RECIPE_PATH = os.path.join(RECIPES_DIR, "tape_presence.json")

RUNTIME_CONFIG_PATH = os.path.join(ROI_DIR, "runtime_config.json")
PRODUCT_PROFILE_PATH = os.path.join(ROI_DIR, "product_profile.json")
TEMPLATE_PATH = os.path.join(ROI_DIR, "align_template.png")

LOGS_ROOT = os.path.join(DATA_DIR, "logs")