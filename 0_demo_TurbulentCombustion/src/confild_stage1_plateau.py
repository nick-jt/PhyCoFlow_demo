"""Stage-B fix 2: continue CoNFiLD stage-1 training to a HELD-OUT codec plateau.

Resumes the unified stage-1 checkpoint (optimizer states + the pending
accumulated decoder gradients are persisted by confild_upstream_training) and
keeps training with upstream semantics -- decoder stepped once per epoch at the
next epoch boundary, latents stepped per batch -- until the held-out codec
rel-L2 stops improving.

Stopping is PATIENCE-based, not wall-budget based: every --eval-every epochs
the codec is scored on --val-offsets held-out snapshots (TUNE-split odd cube-3
indices, never the TEST evens) via a frozen-decoder latent fit; if the mean
z-scored rel-L2 has not improved by --min-delta for --patience consecutive
evaluations, training stops and best_codec-selected + last checkpoints remain.

Checkpoints are the unified stage-1 format, so confild_eval_unified.py loads
them directly.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SRC = ("/home/ntricard/generative_reconstruction/temp/"
       "PhyCoFlow_demo_forked_updated_fpe/0_demo_TurbulentCombustion/src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from confild_upstream_core import (  # noqa: E402
    FieldStatistics,
    PackedJHUCubes,
    batches,
    import_upstream_decoder,
    item_to_snapshot_group,
    octahedral_gather,
)
from confild_upstream_training import (  # noqa: E402
    _append_jsonl,
    _rel_l2_zscore,
    _stage1_checkpoint,
    decode_field,
    fit_latent,
    save_field_figure,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume-ckpt", required=True,
                   help="Stage-1 last.pt (carries optimizer state + pending "
                        "decoder gradients).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data", default="",
                   help="Override data path (e.g. node-local staged copy).")
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=0.002)
    p.add_argument("--val-offsets", type=int, nargs="+", default=[1, 21, 41],
                   help="Offsets into the held-out split (ODD = TUNE, never "
                        "the TEST evens).")
    p.add_argument("--tta-steps", type=int, default=400)
    p.add_argument("--max-epochs", type=int, default=100000)
    p.add_argument("--max-hours", type=float, default=22.0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if any(off % 2 == 0 for off in args.val_offsets):
        raise ValueError("--val-offsets must be ODD (TUNE split); evens are TEST.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stage-1 continuation is GPU-only")
    torch.cuda.set_device(device)

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Evaluation").mkdir(exist_ok=True)

    ck = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
    if int(ck.get("format_version", 0)) < 2 or ck.get("training_stage") != 1:
        raise RuntimeError(f"{args.resume_ckpt} is not a unified stage-1 checkpoint")
    cfg = ck["config"]
    stage = cfg["confild_params"]["stage1"]
    training = stage["training"]
    architecture = ck["architecture"]
    groups = int(ck["groups"])

    data_path = args.data or cfg["shared"]["paths"]["data_path"]
    train_count = int(ck["train_indices"].numel())
    val_count = int(ck["val_indices"].numel())
    dataset = PackedJHUCubes(data_path, train_count=train_count,
                             val_start=train_count, val_count=val_count)
    if not torch.allclose(dataset.coord_min, ck["coord_min"]) or \
       not torch.allclose(dataset.coord_max, ck["coord_max"]):
        raise RuntimeError("coordinate normalisation drift vs checkpoint")
    statistics = FieldStatistics.from_state_dict(ck["field_statistics"])
    torch.save(statistics.state_dict(), run_dir / "field_statistics.pt")

    Decoder = import_upstream_decoder(cfg["confild_params"]["upstream_root"])
    model = Decoder(
        in_coord_features=3,
        in_latent_features=int(architecture["latent_dim"]),
        out_features=dataset.num_fields,
        num_hidden_layers=int(architecture["layers"]),
        hidden_features=int(architecture["hidden_features"]),
    ).to(device)
    model.load_state_dict(ck["decoder"])
    n_items = train_count * groups
    latents = torch.nn.Parameter(torch.zeros(n_items, int(architecture["latent_dim"]),
                                             device=device))
    latents.data.copy_(ck["latents"].to(device))
    decoder_optimizer = torch.optim.Adam(model.parameters(),
                                         lr=float(training["decoder_learning_rate"]))
    latent_optimizer = torch.optim.Adam([latents],
                                        lr=float(training["latent_learning_rate"]))
    decoder_optimizer.load_state_dict(ck["decoder_optimizer"])
    latent_optimizer.load_state_dict(ck["latent_optimizer"])
    for parameter, gradient in zip(model.parameters(), ck.get("decoder_gradients", [])):
        if gradient is not None:
            parameter.grad = gradient.to(device)
    start_epoch = int(ck["epoch"]) + 1
    best_train = float(ck.get("best_train_rel_l2", math.inf))

    snapshots = [statistics.normalize(f) for f in dataset.snapshots(dataset.train_indices)]
    coords = dataset.coords.to(device)
    stats_device = statistics.to(device)
    field_names = list(cfg["shared"]["data"]["field_names"])
    generator = torch.Generator(device="cpu").manual_seed(
        int(cfg["shared"]["seed"]) + start_epoch)

    val_truths = {off: stats_device.normalize(
        dataset.read_snapshot(int(dataset.val_indices[off])).to(device))
        for off in args.val_offsets}

    def codec_score(tag):
        scores = {}
        for off, truth in val_truths.items():
            z = fit_latent(model, coords, truth, int(architecture["latent_dim"]),
                           device, steps=args.tta_steps)
            pred = decode_field(model, z, coords, dataset.num_points, device)
            scores[off] = _rel_l2_zscore(pred, truth, stats_device)
            if tag is not None and off == args.val_offsets[0]:
                save_field_figure(
                    run_dir / "Evaluation" / f"plateau_heldout_off{off}_{tag}.png",
                    dataset.grid_shape, field_names,
                    f"stage-1 continuation - held-out codec fit (val offset {off}) "
                    f"{tag}  rel-L2z {scores[off]:.4f}",
                    truth=truth.cpu().numpy(), prediction=pred.cpu().numpy())
        return scores

    print(f"[plateau] resume epoch {start_epoch} items={n_items} groups={groups} "
          f"val_offsets={args.val_offsets}", flush=True)
    baseline = codec_score(f"ep{start_epoch - 1:05d}")
    best_codec = float(np.mean(list(baseline.values())))
    print(f"[plateau] baseline codec rel-L2z {best_codec:.4f} {baseline}", flush=True)
    since_improve = 0
    started = time.time()
    train_step_time_s = 0.0
    epochs_done = 0

    def save(name, epoch):
        torch.save(_stage1_checkpoint(
            cfg=cfg, model=model, latents=latents,
            decoder_optimizer=decoder_optimizer, latent_optimizer=latent_optimizer,
            epoch=epoch, dataset=dataset, statistics=statistics,
            best_train_rel_l2=best_train,
        ), run_dir / name)

    save("last.pt", start_epoch - 1)
    save("best_codec.pt", start_epoch - 1)

    stop_reason = "max_epochs"
    for epoch in range(start_epoch, args.max_epochs):
        epoch_started = time.time()
        model.train()
        if epoch != 0:
            decoder_optimizer.step()
            decoder_optimizer.zero_grad(set_to_none=True)
        losses = []
        for item_ids_cpu in batches(torch.randperm(n_items, generator=generator),
                                    int(training["batch_size"])):
            point_ids = torch.randint(
                dataset.num_points,
                (item_ids_cpu.numel(), int(training["points_per_item"])),
                generator=generator,
            )
            batch_coords, batch_truth = [], []
            for row, item in enumerate(item_ids_cpu.tolist()):
                snapshot_index, group_index = item_to_snapshot_group(item, groups)
                normalised = snapshots[snapshot_index]
                ids = point_ids[row]
                values = (octahedral_gather(normalised, dataset.grid_shape,
                                            group_index, ids)
                          if group_index else normalised[ids])
                batch_coords.append(coords[ids.to(device)])
                batch_truth.append(values.to(device))
            prediction = model(torch.stack(batch_coords),
                               latents[item_ids_cpu.to(device)].unsqueeze(1))
            loss = F.mse_loss(prediction, torch.stack(batch_truth))
            latent_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            latent_optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize(device)
        epoch_time = time.time() - epoch_started
        train_step_time_s += epoch_time
        epochs_done += 1
        record = {"epoch": epoch, "train_mse": float(np.mean(losses)),
                  "epoch_time_s": epoch_time,
                  "elapsed_s": time.time() - started}

        if (epoch - start_epoch + 1) % args.eval_every == 0:
            scores = codec_score(None)
            score = float(np.mean(list(scores.values())))
            record["heldout_codec_rel_l2_zscore"] = score
            record["heldout_codec_per_snap"] = scores
            improved = score < best_codec - args.min_delta
            if improved:
                best_codec = score
                since_improve = 0
                save("best_codec.pt", epoch)
                codec_score(f"ep{epoch:05d}")  # figure at each improvement
            else:
                since_improve += 1
            save("last.pt", epoch)
            print(f"[plateau] epoch={epoch} codec_rel_l2z={score:.4f} "
                  f"best={best_codec:.4f} since_improve={since_improve}/"
                  f"{args.patience} improved={improved}", flush=True)
            if since_improve >= args.patience:
                stop_reason = "patience"
                _append_jsonl(run_dir / "history.jsonl", record)
                break
        _append_jsonl(run_dir / "history.jsonl", record)
        if (time.time() - started) / 3600.0 >= args.max_hours:
            stop_reason = "wall"
            save("last.pt", epoch)
            break

    summary = {
        "resume_ckpt": args.resume_ckpt,
        "start_epoch": start_epoch,
        "epochs_done": epochs_done,
        "best_codec_rel_l2_zscore": best_codec,
        "stop_reason": stop_reason,
        "train_step_time_s": train_step_time_s,
        "patience": args.patience, "min_delta": args.min_delta,
        "eval_every": args.eval_every, "val_offsets": args.val_offsets,
    }
    json.dump(summary, open(run_dir / "plateau_summary.json", "w"), indent=1)
    print(f"[plateau] DONE {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
