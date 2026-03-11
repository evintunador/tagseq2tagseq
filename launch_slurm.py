"""
Submit a TAGSeq2TAGSeq training run via submitit (SLURM).

Usage:
    python launch_slurm.py [launcher opts] [main.py config overrides]

Launcher options (all optional):
    --nodes N            Number of nodes          (default: 1)
    --gpus-per-node N    GPUs per node            (default: 1)
    --config PATH        Config YAML              (default: configs/baseline.yaml)
    --partition NAME     SLURM partition          (default: compute)
    --time HH:MM:SS      Wall-clock time limit    (default: 24:00:00)
    --mem-per-gpu GB     CPU RAM per GPU in GB    (default: 64)
    --cpus-per-task N    CPU cores per task/GPU   (default: 8)
    --exclude NODES      Comma-separated node list to exclude
    --no-tail            Don't follow logs interactively

Any remaining arguments are forwarded verbatim to main.py / compose_config.

Examples:
    # 1 node, 1 GPU
    python launch_slurm.py \\
        --dataset-dir data/pretokenized_datasets/simplewiki

    # 2 nodes x 4 GPUs each (8 total), 12-hour wall time
    python launch_slurm.py --nodes 2 --gpus-per-node 4 --time 12:00:00 \\
        --dataset-dir data/pretokenized_datasets/simplewiki

    # 8 nodes x 8 GPUs (64 total), no compile
    python launch_slurm.py --nodes 8 --gpus-per-node 8 \\
        --dataset-dir data/pretokenized_datasets/simplewiki \\
        --model.compile false
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Worker -- runs on each SLURM task (one per GPU)
# ---------------------------------------------------------------------------

def _training_worker(main_argv: list, run_dir: str, script: str = 'main') -> None:
    """
    Entry point called by submitit on every SLURM task.

    submitit launches this as a first-class srun task, so each process
    inherits the job's SLURM network context (required for inter-node NCCL
    bootstrap TCP on this cluster).

    SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS are already set by SLURM.
    DistributedManager reads those and initialises the process group.

    Key design note on is_main_process:
      ReproducibilityManager is constructed BEFORE DistributedManager.__enter__()
      runs (both are in the same `with dist_mgr, rep:` statement).  We therefore
      use SLURM_PROCID directly to determine the main rank, rather than
      dist_mgr.is_main (which is only valid after __enter__).
    """
    # ---- env setup (before any torch import) ----
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_DIST_TIMEOUT_SECONDS", "1800")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    # Per-rank compile cache avoids NFS lock contention on first-run compilation.
    job_id  = os.environ.get("SLURM_JOB_ID",  "local")
    proc_id = os.environ.get("SLURM_PROCID",  "0")
    cache   = f"/tmp/torchinductor_{job_id}/rank{proc_id}"
    os.makedirs(cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache

    # ---- reconstruct sys.argv so compose_config sees the right args ----
    sys.argv = ["main"] + main_argv

    # ---- ensure project root is on sys.path ----
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ---- imports (after env is configured) ----
    import argparse as _ap
    from tunalab.configuration import compose_config
    from tunalab.distributed import DistributedManager
    from tunalab.reproducibility import ReproducibilityManager
    import importlib as _il
    _mod = _il.import_module(script)
    train_main = _mod.main

    config = compose_config(_ap.ArgumentParser(add_help=False))

    # SLURM_PROCID is set by srun before our code runs.  Safe to read here.
    is_main_rank = (int(os.environ.get("SLURM_PROCID", "0")) == 0)

    dist_mgr = DistributedManager()
    rep      = ReproducibilityManager(
        output_dir=run_dir,
        is_main_process=is_main_rank,
    )

    with dist_mgr, rep:
        train_main(config, dist_mgr, rep)


# ---------------------------------------------------------------------------
# Launcher -- runs on the login/submit node
# ---------------------------------------------------------------------------

def launch(
    script:        str   = 'main',
    nodes:         int   = 1,
    gpus_per_node: int   = 1,
    config:        str   = "configs/baseline.yaml",
    partition:     str   = "compute",
    time_limit:    str   = "24:00:00",
    mem_per_gpu:   int   = 64,
    cpus_per_task: int   = 8,
    exclude:       str   = None,
    auto_tail:     bool  = True,
    main_argv:     list  = None,
) -> str:
    import submitit

    main_argv = list(main_argv or [])

    # Always inject --config so compose_config has it.
    if "--config" not in main_argv:
        main_argv = ["--config", config] + main_argv

    # Build a timestamped run directory on the shared filesystem.
    project_root = Path(__file__).parent
    run_id  = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = project_root / "runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    slurm_logs_dir = run_dir / ".slurm"
    slurm_logs_dir.mkdir(exist_ok=True)

    executor = submitit.AutoExecutor(
        folder=str(slurm_logs_dir),
        slurm_max_num_timeout=30,
    )

    parts       = time_limit.split(":")
    timeout_min = int(parts[0]) * 60 + int(parts[1])
    if len(parts) > 2:
        timeout_min += int(parts[2]) // 60

    update_params = dict(
        nodes=nodes,
        tasks_per_node=gpus_per_node,   # one srun task per GPU
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_per_gpu * gpus_per_node,
        timeout_min=timeout_min,
        slurm_partition=partition,
        name=f"ts2ts_{run_id}",
    )
    if exclude:
        update_params["slurm_exclude"] = exclude

    executor.update_parameters(**update_params)

    job = executor.submit(_training_worker, main_argv, str(run_dir), script)

    # submitit names log files <job_id>_<task>_log.{out,err}
    stdout_src = slurm_logs_dir / f"{job.job_id}_0_log.out"
    stderr_src = slurm_logs_dir / f"{job.job_id}_0_log.err"
    stdout_lnk = run_dir / "logs" / "stdout.txt"
    stderr_lnk = run_dir / "logs" / "stderr.txt"

    for _ in range(25):
        if stdout_src.exists():
            break
        time.sleep(0.2)

    if stdout_src.exists() and not stdout_lnk.exists():
        stdout_lnk.symlink_to(Path("..") / ".slurm" / stdout_src.name)
    if stderr_src.exists() and not stderr_lnk.exists():
        stderr_lnk.symlink_to(Path("..") / ".slurm" / stderr_src.name)

    print(f"[launch_slurm] Job ID      : {job.job_id}")
    print(f"[launch_slurm] Run dir     : {run_dir}")
    print(f"[launch_slurm] Nodes       : {nodes}  x  {gpus_per_node} GPU(s)  =  {nodes * gpus_per_node} total")
    print(f"[launch_slurm] Logs        : {run_dir / 'logs'}")
    print()

    if auto_tail:
        print("[launch_slurm] Following logs (Ctrl-C stops viewing; job keeps running)...\n")
        try:
            subprocess.run(
                ["tail", "-f", str(stdout_lnk), str(stderr_lnk)],
                check=False,
            )
        except KeyboardInterrupt:
            print("\n[launch_slurm] Detached from logs.")

    return str(job.job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Submit a TAGSeq2TAGSeq training job via submitit/SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--script",        type=str, default="main",
                        help="Python module to run (must expose main())")
    parser.add_argument("--nodes",         type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=1,
                        help="GPUs per node")
    parser.add_argument("--config",        type=str, default="configs/baseline.yaml",
                        help="Config YAML path (relative to project root)")
    parser.add_argument("--partition",     type=str, default="compute",
                        help="SLURM partition")
    parser.add_argument("--time",          type=str, default="24:00:00",
                        help="Wall-clock limit HH:MM:SS")
    parser.add_argument("--mem-per-gpu",   type=int, default=64,
                        help="CPU RAM (GB) per GPU")
    parser.add_argument("--cpus-per-task", type=int, default=8,
                        help="CPU cores per task/GPU")
    parser.add_argument("--exclude",       type=str, default=None,
                        help="Comma-separated nodes to exclude")
    parser.add_argument("--no-tail",       action="store_true",
                        help="Don't auto-tail logs")

    args, main_argv = parser.parse_known_args()

    launch(
        script=args.script,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        config=args.config,
        partition=args.partition,
        time_limit=args.time,
        mem_per_gpu=args.mem_per_gpu,
        cpus_per_task=args.cpus_per_task,
        exclude=args.exclude,
        auto_tail=not args.no_tail,
        main_argv=main_argv,
    )


if __name__ == "__main__":
    main()
