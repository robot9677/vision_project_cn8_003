import json


def load_product_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modules = {}

    if "modules" in data:
        modules = data["modules"]
    elif "enable_modules" in data:
        modules = data["enable_modules"]

    data["modules"] = modules
    data["recipe_name"] = data.get("recipe_name", "tape_presence")

    return data