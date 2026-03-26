def decide_overall(*, recipe, results, auto_mode=False):
    decision = (recipe.get("decision") or {})
    mode = (decision.get("mode") or "any_fail_is_ng").strip().lower()

    oks = [r.ok for r in results.values()]
    if not oks:
        overall_ok = False
    else:
        if mode == "any_fail_is_ng":
            overall_ok = all(oks)
        elif mode == "majority_ok":
            overall_ok = (sum(1 for v in oks if v) >= (len(oks) / 2))
        elif mode == "allow_fail_count":
            max_fail = int(decision.get("max_fail", 0))
            fail_cnt = sum(1 for v in oks if not v)
            overall_ok = (fail_cnt <= max_fail)
        else:
            overall_ok = all(oks)

    if not auto_mode:
        print(f"[DBG] overall decision by recipe : {mode}")

    return overall_ok