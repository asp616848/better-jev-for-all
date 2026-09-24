"""
Real, evidence-producing validation of `RoutingDecoderModel` (PRD.md 5.2b /
14 Q4) against the known-stable `checkpoints/ekvachan-decoder-qwen-benchcorpus`
checkpoint. Run once GPU headroom actually allows it (dev-guidelines rule 4:
check `nvidia-smi` / `ps aux` before loading any model) -- this is not a
pytest unit test, it needs the real checkpoint and a real GPU forward pass.
See `sdk/python/tests/test_client.py` for the mocked-model wire-contract
tests that run everywhere else.

Explicitly passes `text_adapter="benchcorpus"` to `load_default_router` --
bypassing the shipped `EKVACHAN_TEXT_ADAPTER` default, which is
intentionally left as an unset placeholder pending PRD.md 5.2b's
text-vs-vision regression check (see `serve/inference.py`'s module
docstring). This script exercises the checkpoint that has always been the
known-good default, without pretending this override is now the production
answer to that open question.

Three things checked, each written into the manifest with real numbers, not
summarized in prose (dev-guidelines rule 10):

  1. `noul` — is the returned float sane, and does it equal exactly what the
     underlying 2-way `choice("Yes","No")` call itself computes (proving
     `predict_noul` really is "internally a 2-way choice", not a separate
     mechanism)?
  2. `score` — does the probability-weighted position differ from a plain
     argmax-index score, and does reversing `levels`' order change the
     returned `score` in the predictable way index-weighting implies (the
     real check that "never shuffled" is enforced, not just documented)?
  3. `choice`/image routing — only attempted if
     `checkpoints/ekvachan-decoder-qwen-vision/manifest.json` exists yet;
     otherwise recorded as explicitly skipped, with the reason, rather than
     silently omitted.

Run:
    uv run python3 -u -m serve.validate_routing_model
"""

import json
import time
from pathlib import Path

from serve.inference import load_default_router

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    t0 = time.time()
    model = load_default_router(text_adapter="benchcorpus")
    print("loaded:", json.dumps(model.describe(), indent=2))
    report = {"describe": model.describe(), "checks": {}}

    # --- 1. noul: sane float, exactly matches the underlying 2-way choice ---
    noul_cases = [
        (
            "Subject: You've WON $1,000,000!!! Click here NOW to claim your prize before it's gone!!!",
            True,
        ),
        (
            "Subject: Reminder - team meeting tomorrow at 10am in Conference Room B.",
            False,
        ),
    ]
    noul_results = []
    for state, expect_yes in noul_cases:
        p_yes = model.predict_noul(state)
        underlying = model.predict_choice(state, ["Yes", "No"])
        noul_results.append({
            "state": state,
            "p_yes": p_yes,
            "underlying_choice": underlying["choice"],
            "underlying_probabilities": underlying["probabilities"],
            "matches_underlying_choice_p_yes_exactly": p_yes == underlying["probabilities"]["Yes"],
            "expected_answer_is_yes": expect_yes,
            "predicted_answer_is_yes": p_yes > 0.5,
            "correct": (p_yes > 0.5) == expect_yes,
        })
        print("noul:", noul_results[-1])
    report["checks"]["noul"] = noul_results

    # --- 2. score: weighted position != argmax index; order changes result -
    score_state = (
        "My production database is down and customers cannot check out. "
        "Losing $10k/minute."
    )
    levels_ascending = ["low", "medium", "high", "critical"]
    levels_descending = list(reversed(levels_ascending))

    r_asc = model.predict_score(score_state, levels_ascending)
    r_desc = model.predict_score(score_state, levels_descending)
    argmax_level_asc = max(r_asc["probabilities"], key=r_asc["probabilities"].get)
    argmax_level_desc = max(r_desc["probabilities"], key=r_desc["probabilities"].get)
    argmax_idx_asc = levels_ascending.index(argmax_level_asc)

    score_check = {
        "state": score_state,
        "levels_ascending": levels_ascending,
        "result_ascending": r_asc,
        "levels_descending": levels_descending,
        "result_descending": r_desc,
        "argmax_level_name_ascending": argmax_level_asc,
        "argmax_level_name_descending": argmax_level_desc,
        # Content-driven: the model should identify the same semantic level
        # (e.g. "critical") as most likely regardless of which letter/position
        # it's presented at -- that's expected and fine.
        "argmax_level_name_is_order_invariant": argmax_level_asc == argmax_level_desc,
        "argmax_index_in_ascending_order": argmax_idx_asc,
        "weighted_score_differs_from_plain_argmax_index": r_asc["score"] != float(argmax_idx_asc),
        # The load-bearing check: the *numeric* score must respond to the
        # caller's own order (never silently re-sorted back to ascending
        # internally) even while the semantic answer above stays the same.
        "reversing_order_changes_numeric_score": r_asc["score"] != r_desc["score"],
        "probabilities_keyed_by_exact_level_strings_given": (
            set(r_asc["probabilities"]) == set(levels_ascending)
            and set(r_desc["probabilities"]) == set(levels_descending)
        ),
    }
    print("score:", score_check)
    report["checks"]["score"] = score_check

    # --- 3. image routing -- only if the vision checkpoint has landed -------
    vision_manifest = REPO_ROOT / "checkpoints" / "ekvachan-decoder-qwen-vision" / "manifest.json"
    report["vision_checkpoint_ready"] = vision_manifest.exists()
    if vision_manifest.exists():
        report["checks"]["image_routing"] = "vision checkpoint present -- see follow-up manifest for the real check"
    else:
        report["checks"]["image_routing"] = {
            "status": "SKIPPED",
            "reason": f"{vision_manifest} does not exist yet -- PRD.md 13a.11's vision training "
                      "run had not finished as of this validation run.",
        }
        # Still worth proving the routing *rule* itself is enforced even
        # without a real vision checkpoint: an image-bearing request must be
        # rejected with a clear error, not silently answered text-only.
        import base64
        tiny_png_b64 = base64.b64encode(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
                "53de0000000c4944415478da6360000002000155a1eb280000000049454e44ae426082"
            )
        ).decode("ascii")
        try:
            model.predict_choice("a screenshot", ["a", "b"], image_b64=tiny_png_b64)
            routing_enforced = False
            routing_error = None
        except Exception as e:  # noqa: BLE001 - recording whatever it is, not asserting a type here
            routing_enforced = type(e).__name__ == "ChoiceUnsupportedError"
            routing_error = f"{type(e).__name__}: {e}"
        report["checks"]["image_routing_rule_enforced_pre_vision_checkpoint"] = {
            "raised_choice_unsupported_error": routing_enforced,
            "error": routing_error,
        }
        print("image routing rule check:", report["checks"]["image_routing_rule_enforced_pre_vision_checkpoint"])

    report["elapsed_seconds"] = time.time() - t0
    report["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_path = REPO_ROOT / "results" / f"routing-model-validation-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
