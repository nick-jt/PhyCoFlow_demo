"""Training-health watchdog for the S3GM run.

WHY THIS WAS REWRITTEN
The previous version aborted on `obs_rmse_z`, a SAMPLING diagnostic. It killed
job 17001893 at 15,000 optimizer steps while that run was healthy: train loss
0.0265 / val 0.0350 against a zero-score baseline of 1.0 (~97% of the denoising
signal captured). The isolation sweep (jobs 17013707/17013880) then showed the
divergence came from the sampler's guidance weights, not from training:

  alpha=0, beta=0     std 0.42  obs_rmse_z 1.10   <- score/predictor/windows OK
  alpha=0.5, beta=0   std 0.71  obs_rmse_z 0.021  <- DPS stable and helping
  alpha=1.0, beta=0   std 797                     <- DPS stability edge at ~1
  alpha=0, beta=0.4   std 759                     <- upstream consistency weight
                                                     diverges ON ITS OWN

Both alpha and beta are sampler-only. Verified: `beta`/`loss_consis` appear in
neither s3gm3d_loss nor run_epoch_s3gm3d here, and upstream has `loss_consis`
only in sampler/utils.py -- zero hits in train.py and trainer/loss.py.

So a sampling metric must never terminate training. This watchdog aborts ONLY
on a training pathology:
  * non-finite train or val loss                                    -> ABORT
  * train loss still > LOSS_MAX at >= GATE_STEPS optimizer steps    -> ABORT
`obs_rmse_z` is still parsed and logged every poll as a MONITOR, and can no
longer terminate anything.
"""
from __future__ import annotations
import argparse, json, math, re, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--job-id", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gate-steps", type=int, default=15000)
ap.add_argument("--loss-max", type=float, default=0.30,
                help="train loss above this at the gate = broken run "
                     "(0.0265 was achieved by the healthy run; 1.0 = zero score)")
ap.add_argument("--poll-s", type=float, default=120.0)
a = ap.parse_args()

INSTR = re.compile(r"\[instr\] epoch=(\d+).*?opt_steps_total=(\d+)")
DUTY = re.compile(r"data_s=([0-9.]+) compute_s=([0-9.]+)")
TRAIN = re.compile(r"\[train\] epoch=(\d+)\s+loss=([0-9.eE+-]+)")
VAL = re.compile(r"val=([0-9.eE+-]+)")
SAMP = re.compile(r"\[sampler\].*?obs_rmse_z=([0-9.eEnaN+-]+)")


def _f(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def scan(path):
    losses, steps, obs, tc, td = [], 0, [], 0.0, 0.0
    if not path.exists():
        return losses, steps, obs, tc, td
    for raw in open(path, "r", errors="replace"):
        for line in raw.replace("\r", "\n").split("\n"):
            m = INSTR.search(line)
            if m:
                steps = int(m.group(2))
                d = DUTY.search(line)
                if d:
                    td += float(d.group(1)); tc += float(d.group(2))
                continue
            m = TRAIN.search(line)
            if m:
                v = VAL.search(line)
                losses.append({"epoch": int(m.group(1)), "steps": steps,
                               "train": _f(m.group(2)),
                               "val": _f(v.group(1)) if v else None})
                continue
            m = SAMP.search(line)
            if m:
                obs.append({"steps": steps, "obs_rmse_z": _f(m.group(1))})
    return losses, steps, obs, tc, td


def alive(j):
    return bool(subprocess.run(["squeue", "-j", j, "-h", "-o", "%T"],
                               capture_output=True, text=True).stdout.strip())


def elapsed(j):
    r = subprocess.run(["sacct", "-j", j, "--format=ElapsedRaw", "-n", "-P"],
                       capture_output=True, text=True).stdout.split("\n")[0].strip()
    return float(r) if r.isdigit() else 0.0


def finish(verdict, reason, payload):
    payload.update({"verdict": verdict, "reason": reason, "criterion": __doc__})
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"[verdict] {verdict}: {reason}", flush=True)
    raise SystemExit(0)


log, gated = Path(a.log), False
print(f"[watchdog] TRAINING-HEALTH ONLY. log={log} job={a.job_id} "
      f"gate_steps={a.gate_steps} loss_max={a.loss_max} "
      f"(obs_rmse_z is monitored, never fatal)", flush=True)

while True:
    losses, steps, obs, tc, td = scan(log)
    el = elapsed(a.job_id)
    payload = {"job_id": a.job_id, "opt_steps": steps,
               "loss_history": losses[-40:], "obs_rmse_z_monitor": obs,
               "duty": {"compute_s": tc, "loader_wait_s": td,
                        "slurm_elapsed_s": el,
                        "epoch_duty": (tc / (tc + td)) if tc + td > 0 else None,
                        "job_duty": (tc / el) if el > 0 else None}}

    for L in losses:
        bad = [k for k in ("train", "val")
               if L[k] is not None and not math.isfinite(L[k])]
        if bad:
            subprocess.run(["scancel", a.job_id])
            finish("ABORT", f"non-finite {'/'.join(bad)} loss at epoch {L['epoch']}",
                   payload)

    if losses and not gated and losses[-1]["steps"] >= a.gate_steps:
        gated = True
        tl = losses[-1]["train"]
        print(f"[watchdog] training gate at {losses[-1]['steps']} steps: "
              f"train_loss={tl:.5f} (max allowed {a.loss_max})", flush=True)
        if tl > a.loss_max:
            subprocess.run(["scancel", a.job_id])
            finish("ABORT", f"train loss {tl:.5f} > {a.loss_max} at "
                            f"{losses[-1]['steps']} steps; run is not learning", payload)

    last = losses[-1] if losses else None
    ob = obs[-1]["obs_rmse_z"] if obs else float("nan")
    print(f"[watchdog] steps={steps} "
          + (f"epoch={last['epoch']} train={last['train']:.5f} " if last else "")
          + (f"val={last['val']:.5f} " if last and last["val"] is not None else "")
          + f"| monitor obs_rmse_z={ob:.3e} (non-fatal) "
          + f"job_duty={payload['duty']['job_duty']}", flush=True)

    if not alive(a.job_id):
        finish("PASS", "training job ended; no training pathology detected", payload)
    time.sleep(a.poll_s)
