def execute_command(app, cmd, frame_gray8, vis_bgr):
    st = app.state

    if cmd == app.UICmd.QUIT:
        st.quit_requested = True
        return

    if cmd == app.UICmd.TOGGLE_MODE:
        app._toggle_mode()
        return

    if cmd == app.UICmd.TOGGLE_AUTO_INSPECT:
        if st.edit_mode:
            st.status = "AUTO INSPECT only in RUN"
        else:
            app._toggle_auto_inspect()
        return

    if st.edit_mode:
        _execute_edit_mode(app, cmd, frame_gray8)
    else:
        _execute_run_mode(app, cmd, frame_gray8, vis_bgr)


def _execute_edit_mode(app, cmd, frame_gray8):
    st = app.state

    if cmd == app.UICmd.SAVE:
        app._save_roi_and_template(frame_gray8)
        return

    if cmd == app.UICmd.NEXT:
        try:
            app.roi_mgr.select_next()
            app.editor.on_select_changed()
            st.status = f"Selected ROI: {app.roi_mgr.selected_id}"
        except Exception:
            st.status = "Select next failed"
        return

    if cmd == app.UICmd.CLEAR:
        try:
            if hasattr(app.roi_mgr, "clear"):
                app.roi_mgr.clear()
            else:
                for r in list(app.roi_mgr.list()):
                    try:
                        app.roi_mgr.remove(r["id"])
                    except Exception:
                        pass
            st.status = "Cleared ROIs"
        except Exception:
            st.status = "Clear failed"
        return

    if cmd == app.UICmd.DELETE:
        try:
            sid = app.roi_mgr.selected_id
            if sid is not None:
                app.roi_mgr.remove(sid)
                app.editor.on_select_changed()
                st.status = f"Deleted ROI {sid}"
        except Exception:
            st.status = "Delete failed"
        return


def _execute_run_mode(app, cmd, frame_gray8, vis_bgr):
    st = app.state

    if cmd == app.UICmd.INSPECT:
        app._inspect_once(frame_gray8, vis_bgr)
        return

    if cmd == app.UICmd.AUTOTUNE:
        try:
            app.inspector.autotune_recipe_from_frame(frame_gray8, save_path=app.recipe_path)
            st.status = "Autotune done"
        except Exception:
            st.status = "Autotune failed"
        return

    if cmd == app.UICmd.RELOAD:
        try:
            app.inspector.reload_recipe()
            st.status = "Recipe reloaded"
        except Exception:
            st.status = "Reload failed"
        return