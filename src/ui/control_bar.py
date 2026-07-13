from ui.overlay_clean import draw_control_bar


def build_control_buttons(edit_mode: bool, show_service: bool = True):
    if edit_mode:
        buttons = [
            {"id": "toggle_edit", "label": "RUN", "color": (70, 130, 180), "enabled": True},
            {"id": "save", "label": "SAVE", "color": (120, 0, 120), "enabled": True},
            {"id": "next", "label": "NEXT", "color": (80, 80, 80), "enabled": True},
            {"id": "delete", "label": "DELETE", "color": (40, 40, 120), "enabled": True},
            {"id": "clear", "label": "CLEAR", "color": (40, 0, 80), "enabled": True},
        ]
    else:
        buttons = [
            {"id": "toggle_edit", "label": "EDIT", "color": (70, 130, 180), "enabled": True},
            {"id": "inspect", "label": "INSPECT", "color": (0, 120, 0), "enabled": True},
            {"id": "autoinspect", "label": "AUTO", "color": (80, 80, 80), "enabled": True},
            {"id": "autotune", "label": "AUTOTUNE", "color": (0, 120, 120), "enabled": True},
            {"id": "reload", "label": "RELOAD", "color": (120, 120, 0), "enabled": True},
        ]

    if show_service:
        buttons.append(
            {"id": "service", "label": "SERVICE", "color": (70, 70, 20), "enabled": True}
        )

    buttons.append(
        {"id": "quit", "label": "QUIT", "color": (120, 0, 0), "enabled": True}
    )
    return buttons


def render_control_bar(vis, edit_mode: bool, show_service: bool = True):
    return draw_control_bar(vis, build_control_buttons(edit_mode, show_service=show_service))


BUTTON_ID_TO_CMD_NAME = {
    "toggle_edit": "TOGGLE_MODE",
    "inspect": "INSPECT",
    "autotune": "AUTOTUNE",
    "reload": "RELOAD",
    "save": "SAVE",
    "next": "NEXT",
    "nxt": "NEXT",
    "clear": "CLEAR",
    "delete": "DELETE",
    "quit": "QUIT",
    "autoinspect": "TOGGLE_AUTO_INSPECT",
    "service": "TOGGLE_SERVICE_PANEL",
}


KEY_TO_CMD_NAME = {
    1113943: "QUIT",   # END
    27: "QUIT",        # ESC
    ord("q"): "QUIT",
    ord("e"): "TOGGLE_MODE",
    ord("s"): "SAVE",
    ord("n"): "NEXT",
    ord("r"): "CLEAR",
    ord("p"): "RELOAD",
    ord("c"): "AUTOTUNE",
    32: "INSPECT",
    ord("x"): "DELETE",
    ord("a"): "TOGGLE_AUTO_INSPECT",
}


def key_to_cmd(key: int, cmd_enum_cls):
    cmd_name = KEY_TO_CMD_NAME.get(key)
    if cmd_name is None:
        return cmd_enum_cls.NONE
    return getattr(cmd_enum_cls, cmd_name, cmd_enum_cls.NONE)



def button_id_to_cmd(button_id: str, cmd_enum_cls):
    cmd_name = BUTTON_ID_TO_CMD_NAME.get(button_id)
    if cmd_name is None:
        return cmd_enum_cls.NONE
    return getattr(cmd_enum_cls, cmd_name, cmd_enum_cls.NONE)
