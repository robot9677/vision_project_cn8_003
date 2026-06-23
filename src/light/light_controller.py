class BaseLightController:
    def start(self):
        pass

    def stop(self):
        pass

    def set_brightness(self, light_id, brightness):
        pass

    def get_state(self):
        return {}


class NullLightController(BaseLightController):
    def start(self):
        print("[LIGHT] disabled")

    def stop(self):
        pass


class MockLightController(BaseLightController):
    def __init__(self, light_cfg):
        self.light_cfg = light_cfg or {}
        self.lights = self.light_cfg.get("lights", [])
        self.state = {}

        for item in self.lights:
            light_id = item.get("id")
            if not light_id:
                continue

            self.state[light_id] = {
                "camera_id": item.get("camera_id"),
                "brightness": int(item.get("brightness", 0)),
            }

    def start(self):
        print(f"[LIGHT] mock started channels={len(self.state)}")

        for light_id, info in self.state.items():
            print(
                f"[LIGHT] {light_id} "
                f"camera={info.get('camera_id')} "
                f"brightness={info.get('brightness')}"
            )

    def stop(self):
        print("[LIGHT] mock stopped")

    def set_brightness(self, light_id, brightness):
        if light_id not in self.state:
            print(f"[LIGHT] unknown light_id: {light_id}")
            return False

        brightness = max(0, min(100, int(brightness)))
        self.state[light_id]["brightness"] = brightness

        print(f"[LIGHT] mock set {light_id} brightness={brightness}")
        return True

    def get_state(self):
        return dict(self.state)


def create_light_controller_from_hardware_config(hardware_cfg):
    hardware_cfg = hardware_cfg or {}

    light_sets = hardware_cfg.get("light_sets", {})
    active_light_set = hardware_cfg.get("active_light_set")

    if not active_light_set:
        return NullLightController()

    light_cfg = light_sets.get(active_light_set)

    if not light_cfg:
        print(f"[LIGHT] active light set not found: {active_light_set}")
        return NullLightController()

    backend = light_cfg.get("backend", "none")

    if backend == "mock":
        return MockLightController(light_cfg)

    if backend == "none":
        return NullLightController()

    print(f"[LIGHT] unsupported backend: {backend}")
    return NullLightController()