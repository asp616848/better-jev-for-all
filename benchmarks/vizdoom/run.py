"""
CLI entry point for the ViZDoom episodic harness (PRD.md 8.1 item 2 / 8.1d).
See `benchmarks/vizdoom/README.md` for what this proves and does not yet
prove, and `benchmarks/common/episodic.py` for the shared closed-loop runner.

Usage:
  # The real text-arm run, once a wide-schema/benchcorpus checkpoint exists
  # (PRD.md 8.1d checklist step 6 -- this is THE milestone comparison against
  # Von's published 9.00 kills / 12.11s survival):
  python -m benchmarks.vizdoom.run --backend decoder-multischema \\
      --checkpoint-dir checkpoints/ekvachan-decoder-qwen-benchcorpus \\
      --scenario both --rubric both

  # The rubric-oracle baseline (PRD.md 8.1d checklist step 3) -- no model, no
  # GPU, no checkpoint, real ViZDoom episodes:
  python -m benchmarks.vizdoom.run --policy oracle --scenario both

  # The random-policy baseline (PRD.md 8.1d Finding 2), matching this
  # session's own validation methodology (8 seeds x 10 episodes each,
  # tics=4) rather than the 1-episode-per-seed real-comparison protocol:
  python -m benchmarks.vizdoom.run --backend random --scenario both \\
      --seeds 0,1,2,3,4,5,6,7 --episodes-per-seed 10

  # Harness wiring self-test -- real ViZDoom episodes, a non-trained
  # keyword-overlap backend, no torch/transformers/peft/checkpoint:
  python -m benchmarks.vizdoom.run --backend decoder-multischema-mock --scenario both

  # With DAgger-teacher data emission (PRD.md 8.1d Finding 4 point 3):
  python -m benchmarks.vizdoom.run --backend decoder-multischema-mock \\
      --dagger-out training/dagger_data/vizdoom
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vizdoom as vzd

from benchmarks.common.episodic import BackendPolicy, DaggerSink, OraclePolicy, ScenarioSpec, run_episodic_harness
from benchmarks.common.harness import build_backend, make_arg_parser
from benchmarks.vizdoom import env as vzd_env
from benchmarks.vizdoom import observe, rubric

SCENARIOS = ("defend_the_center", "health_gathering")


def _make_scenario_spec(scenario_name: str) -> ScenarioSpec:
    return ScenarioSpec(
        name=scenario_name,
        seeds=list(vzd_env.SCENARIO_SEEDS[scenario_name]),
        metric_name=vzd_env.SCENARIO_METRIC_NAME[scenario_name],
        env_factory=lambda name, seed, capture_frames: vzd_env.DoomEnvironment(name, seed, capture_frames),
        observe_fn=observe.observe_text,
        rubric_action_fn=rubric.rubric_action,
        extract_metric_fn=vzd_env.SCENARIO_METRIC_FN[scenario_name],
    )


def main() -> list[dict]:
    parser = make_arg_parser("benchmarks.vizdoom.run")
    parser.add_argument(
        "--scenario", choices=[*SCENARIOS, "both"], default="both",
        help="which ViZDoom scenario to run. 'both' runs defend_the_center then health_gathering.",
    )
    parser.add_argument(
        "--rubric", choices=[*rubric.RUBRIC_MODES, "both"], default="both",
        help="'von' = Von's own instructions/criteria verbatim (the headline like-for-like "
             "number, comparable to 9.00 kills / 5.62 kills). 'none' = a bare task description, "
             "no if/then rule. 'both' (default) runs and publishes both -- PRD.md 8.1d Finding 3: "
             "publishing only one hides how much of the score is prompt vs. game understanding.",
    )
    parser.add_argument(
        "--arm", choices=["text"], default="text",
        help="observation modality. Only 'text' is implemented -- the vision arm (PRD.md 8.1d "
             "step 8) is scoped out of this build; see observe.py::observe_frame's docstring.",
    )
    parser.add_argument(
        "--policy", choices=["model", "oracle"], default="model",
        help="'model' drives the episode with --backend (a real ChoiceBackend). 'oracle' ignores "
             "--backend entirely and runs the rubric's own deterministic rule as the policy -- "
             "PRD.md 8.1d checklist step 3's zero-cost, no-model, no-GPU baseline.",
    )
    parser.add_argument("--tics", type=int, default=4, help="frameskip -- Von's own published value.")
    parser.add_argument("--step-cap", type=int, default=300, help="max model decisions per episode -- Von's own published value.")
    parser.add_argument(
        "--episodes-per-seed", type=int, default=1,
        help="1 (default) matches Von's own real-comparison protocol (one episode per published "
             "seed). Set higher (e.g. 10) only for a random/oracle-policy baseline measurement "
             "that needs within-seed variance, matching this session's own validation methodology "
             "-- see benchmarks/vizdoom/README.md.",
    )
    parser.add_argument(
        "--seeds", default=None,
        help="comma-separated int seeds overriding the scenario's published 8-seed list (e.g. "
             "for a random-baseline sweep on 0..7). Default: use the published Von seeds "
             "(benchmarks/vizdoom/env.py's DEFEND_SEEDS/HEALTH_SEEDS) -- required for any run "
             "claiming comparability to Von's published numbers.",
    )
    parser.add_argument(
        "--dagger-out", default=None,
        help="if set, emit on-policy (text, frame, action) DAgger-teacher triples under this "
             "directory (PRD.md 8.1d Finding 4 point 3) -- see benchmarks/common/episodic.py's "
             "DaggerSink docstring for the storage format. Off by default: this is real, "
             "regenerable output for future consumption, not needed for a benchmark score.",
    )
    args = parser.parse_args()

    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    scenario_names = list(SCENARIOS) if args.scenario == "both" else [args.scenario]
    rubric_modes = list(rubric.RUBRIC_MODES) if args.rubric == "both" else [args.rubric]

    if args.policy == "model":
        backend = build_backend(args)
        policy_for = lambda scenario_name: BackendPolicy(backend)  # noqa: E731 -- same backend, every scenario/rubric
    else:
        policy_for = lambda scenario_name: OraclePolicy(rubric.rubric_action, scenario_name)  # noqa: E731

    session_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dagger_root = Path(args.dagger_out) / session_ts if args.dagger_out else None

    all_results: list[dict] = []
    for scenario_name in scenario_names:
        spec = _make_scenario_spec(scenario_name)
        policy = policy_for(scenario_name)
        for rmode in rubric_modes:
            dagger_sink = None
            if dagger_root is not None:
                dagger_sink = DaggerSink(dagger_root / f"{scenario_name}_{rmode}")

            provenance = {
                "source": "live ViZDoom simulation, not a static dataset -- vendored protocol "
                          "constants only (Apache-2.0, from wfzyx/von@master's benchmarks/"
                          "doom_eval.py + run_doom_benchmark.py, fetched via the GitHub API "
                          "2026-09-24; see NOTICE.md)",
                "vizdoom_version": vzd.__version__,
                "scenario_cfg": f"vizdoom.scenarios_path/{scenario_name}.cfg (stock, ships in the wheel)",
                "seeds": seeds_override or vzd_env.SCENARIO_SEEDS[scenario_name],
                "seeds_are_vons_published_seeds": seeds_override is None,
                "tics": args.tics,
                "step_cap": args.step_cap,
                "rubric_mode": rmode,
                "arm": args.arm,
            }

            result = run_episodic_harness(
                benchmark="vizdoom",
                args=args,
                scenario=spec,
                rubric=rmode,
                get_choice_fn=rubric.get_choice,
                policy=policy,
                tics=args.tics,
                step_cap=args.step_cap,
                episodes_per_seed=args.episodes_per_seed,
                seeds_override=seeds_override,
                dataset_provenance=provenance,
                dagger_sink=dagger_sink,
            )
            all_results.append(result)

    return all_results


if __name__ == "__main__":
    main()
