import os, torch, torch.distributed as dist
from datetime import timedelta
rank=int(os.environ["SLURM_PROCID"]); lr=int(os.environ["SLURM_LOCALID"]); ws=int(os.environ["SLURM_NTASKS"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", rank=rank, world_size=ws, device_id=torch.device("cuda",lr), timeout=timedelta(seconds=120))
print(f"[rank{rank}] nccl pg init OK", flush=True)
g = dist.new_group(backend="gloo")   # the secondary Gloo group DistributedManager creates
print(f"[rank{rank}] gloo new_group OK", flush=True)
dist.barrier(group=g)                 # cpu_barrier uses this
print(f"[rank{rank}] gloo barrier OK", flush=True)
t=torch.ones(1024,device=f"cuda:{lr}"); dist.all_reduce(t); torch.cuda.synchronize()
print(f"[rank{rank}] nccl all_reduce OK sum={t.sum().item()}", flush=True)
dist.destroy_process_group()
