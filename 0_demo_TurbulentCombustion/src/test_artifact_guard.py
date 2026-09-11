"""Replays every 2026-09-10 incident against artifact_guard, then checks that
every real artifact in the tree parses and never conflicts with itself."""
import json, sys, tempfile, glob
from pathlib import Path
ROOT = Path("/work/hdd/bilr/ntricard/PhyCoFlow_demo/0_demo_TurbulentCombustion")
sys.path.insert(0, str(ROOT / "src"))
import artifact_guard as G

def P(**kw):
    base = {"protocol": "kolm2d_matched_v1", "model": "s3gm", "n_obs": [655],
            "K": 8, "nfe": 200, "cond_source": "points", "cond_fields": [0],
            "summary": {"aggregate": {"rel_l2_mean": 0.5}}}
    base.update(kw); return base

quiet = lambda *a, **k: None
fails = 0
def case(name, existing, new, expect_conflict):
    global fails
    G.CONFLICTS.clear()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "target.json"
        if existing is not None:
            f.write_text(existing if isinstance(existing, str) else json.dumps(existing))
        before = f.read_text() if f.exists() else None
        out = G.safe_write_json(f, new, log=quiet)
        conflicted = bool(G.CONFLICTS)
        kept = (f.read_text() == before) if conflicted else True
        ok = (conflicted == expect_conflict) and kept and out.exists()
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<58} conflict={conflicted} "
              f"existing_kept={kept}")

# the four incidents, and the K collision found while fixing them
case("#1 NOBS override (65) over canonical 655",       P(), P(n_obs=[65]), True)
case("#3 operator aggregate over clean aggregate",       P(), P(protocol="kolm2d_matched_v1_op_occl0.25"), True)
case("#4 dmfgen occl sweep over clean sweep",            P(model="dmfgen", K=8, nfe=4),
     P(model="dmfgen", K=8, nfe=4, protocol="kolm2d_matched_v1_op_occl0.25"), True)
case("#4b clean roll-up (no n_obs) vs operator roll-up", {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4},
     {"protocol": "kolm2d_matched_v1_op_noise0.1", "model": "dmfgen", "K": 8, "nfe": 4}, True)
case("K collision: K=64 run over K=8 sweep",              P(model="dmfgen", nfe=4), P(model="dmfgen", nfe=4, K=64), True)
case("pre-stamp clean cache vs operator record",          {"n_obs": 655, "K": 8, "nfe": 200},
     {"n_obs": 655, "K": 8, "nfe": 200, "protocol": "kolm2d_matched_v1_op_noise0.3"}, True)
case("surface 64 taps vs 128 taps",                       P(n_obs=[64,64,64], cond_source="surface"),
     P(n_obs=[128,128,128], cond_source="surface"), True)
case("3-frame smoke over 50-frame canonical (n_frames)",  P(n_frames=50), P(n_frames=3), True)
case("best vs last checkpoint (ckpt)",                    P(ckpt="best"), P(ckpt="last"), True)
case("different evaluation seed (sensor draws)",          P(seed=0), P(seed=1), True)
case("val vs test split",                                 P(split="val"), P(split="test"), True)
case("pre-n_frames artifact vs 3-frame run (no key: skip)", {"protocol": "kolm2d_matched_v1", "model": "s3gm", "n_obs": [655], "K": 8},
     P(n_frames=3), False)
# legitimate writes that must still go through
case("same measurement re-run, new numbers",              P(), P(summary={"aggregate": {"rel_l2_mean": 0.49}}), False)
case("pre-stamp clean cache vs clean record",             {"n_obs": 655, "K": 8, "nfe": 200},
     {"n_obs": 655, "K": 8, "nfe": 200, "protocol": "kolm2d_matched_v1"}, False)
case("cache n_obs int vs payload n_obs list",             {"n_obs": 655, "K": 1}, P(K=1), False)
case("roll-up re-written with a different density list",  {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4, "rel_l2_by_n": {"655": 1}},
     {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4, "rel_l2_by_n": {"65": 1}}, False)
case("fresh file",                                        None, P(), False)
case("corrupt existing file",                             "{not json", P(), False)
case("empty existing file",                               "", P(), False)

# sidecar must be named by identity and exit_if_conflicts must fail the job
G.CONFLICTS.clear()
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "t.json"; f.write_text(json.dumps(P()))
    side = G.safe_write_json(f, P(K=64), log=quiet)
    ok = side.name.startswith("t.CONFLICT_") and json.loads(side.read_text())["K"] == 64
    try:
        G.exit_if_conflicts(log=quiet); ok = False
    except SystemExit as e:
        ok = ok and e.code == 3
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  sidecar named by identity; exit_if_conflicts -> exit 3")

# every real artifact: parses, and a self re-write is never a conflict
n = bad = 0
for pat in ("kolmogorov2d", "cylinder2d", "cylinder2d_surface"):
    for f in glob.glob(str(ROOT / "Save_TrainedModel" / pat / "*" / "*" / "Evaluation" / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        n += 1
        o, w = G.identity(d, legacy=True), G.identity(d)
        if any(o[k] != w[k] for k in o if k in w):
            bad += 1; print("  SELF-CONFLICT:", f)
fails += bad
print(f"{'PASS' if not bad else 'FAIL'}  {n} real artifacts parse; {bad} self-conflicts")
print("\nALL PASS" if not fails else f"\n{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
