import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1 << 20, device="cuda") * (r + 1)
dist.all_reduce(x); torch.cuda.synchronize()
if r == 0: print("ALLREDUCE_OK", x[0].item())
dist.destroy_process_group()
