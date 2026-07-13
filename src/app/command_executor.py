from inspection.inspect_service import run_inspect_once

def execute_command(app, cmd, frame_gray8, vis_bgr):
    st = app.state

    if cmd == app.UICmd.QUIT:
        st.quit_requested = True
        return

    if cmd == app.UICmd.TOGGLE_MODE:
        app._toggle_mode()
        return

    if cmd == app.UICmd.TOGGLE_SERVICE_PANEL:
        app._toggle_service_panel()
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
        if frame_gray8 is None:
            st.status = "SAVE BLOCKED: NO CAMERA FRAME"
            return
        app._save_roi_and_template(frame_gray8)
        return

    if cmd == app.UICmd.NEXT:
        try:
            app.roi_mgr.select_next()
            app.editor.on_select_changed()
           # st.status = f"Selected ROI: {app.roi_mgr.selected_id}"
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
        if frame_gray8 is None:
            st.status = "INSPECT BLOCKED: NO CAMERA FRAME"
            return

        spot_cfg = app._get_spot_light_cfg() if hasattr(app, "_get_spot_light_cfg") else {}
        spot_enabled = bool(spot_cfg.get("enabled", False))

        print(
            f"[CMD INSPECT] spot_enabled={spot_enabled} "
            f"armed={bool(getattr(app.state, 'spot_armed', False))}"
        )

        if spot_enabled and hasattr(app, "_spot_prearm") and hasattr(app, "_run_spot_inspect_once"):
            if not bool(getattr(app.state, "spot_armed", False)):
                print("[CMD INSPECT] FIRST_CLICK_PREARM_ONLY")
                app._spot_prearm(trigger="MANUAL")
                app.state.status = "MANUAL READY: PRESS INSPECT AGAIN"
                return

            print("[CMD INSPECT] SECOND_CLICK_RUN_INSPECT")

            app._run_spot_inspect_once(
                frame_gray8,
                vis_bgr,
                avg5=True,
                trigger="MANUAL",
            )
            return

        run_inspect_once(
            cam=app.cam,
            inspector=app.inspector,
            runtime_cfg=app.runtime_cfg,
            state=app.state,
            frame_gray8=frame_gray8,
            vis_bgr=vis_bgr,
            avg5=True,
            use_cache=False,
        )
        return

    if cmd == app.UICmd.AUTOTUNE:
        if frame_gray8 is None:
            st.status = "AUTOTUNE BLOCKED: NO CAMERA FRAME"
            return
        try:
            app.inspector.autotune_recipe_from_frame(frame_gray8)
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