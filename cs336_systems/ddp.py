import argparse
from dataclasses import dataclass, field
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.optimizer_state_sharding import ShardedOptimizer

class ToyModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear(x)))

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param is not None:
                dist.broadcast(param.data, src=0)

        def make_hook(_):
            def hook_fn(tensor: torch.Tensor):
                if tensor.grad is not None:
                    handle = dist.all_reduce(tensor.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append(handle)
            return hook_fn

        self.handles = []
        for param in self.module.parameters():
            if param is not None and param.requires_grad:
                param.register_post_accumulate_grad_hook(make_hook(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size
        self.handles.clear()

@dataclass
class Bucket:
    params: list[torch.nn.Parameter] = field(default_factory=list)
    size: int = 0
    cnt_processed_params: int = 0

@dataclass
class Handle:
    handle: dist.Work
    flattened_grads: torch.Tensor
    grads: list[torch.Tensor]
    bucket_idx: int

class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        for param in self.module.parameters():
            if param is not None:
                dist.broadcast(param.data, src=0)

        self.buckets: list[Bucket] = []
        self.init_backets()

        self.handles: list[Handle] = []
        for i, bucket in enumerate(self.buckets):
            for param in bucket.params:
                param.register_post_accumulate_grad_hook(self.make_hook(i))

    def init_backets(self):        
        for param in list(self.module.parameters())[::-1]:
            if param is not None and param.requires_grad:
                num_param = param.numel()
                param_size = num_param*param.element_size()
                if len(self.buckets)==0 or self.buckets[-1].size+param_size>self.bucket_size_bytes:
                    self.buckets.append(Bucket())
                self.buckets[-1].params.append(param)
                self.buckets[-1].size += param_size

    def make_hook(self, bucket_idx: int):
        def hook_fn(_):
            self.buckets[bucket_idx].cnt_processed_params += 1
            if self.buckets[bucket_idx].cnt_processed_params == len(self.buckets[bucket_idx].params):
                grads = [param.grad for param in self.buckets[bucket_idx].params]
                if len(grads) == 0:
                    return
                flattened_grads = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                
                self.handles.append(Handle(handle, flattened_grads, grads, bucket_idx))
                self.buckets[bucket_idx].cnt_processed_params = 0
        return hook_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.handle.wait()

            handle.flattened_grads.div_(self.world_size)
            unflattened_grads = torch._utils._unflatten_dense_tensors(handle.flattened_grads, handle.grads)

            for old_param, new_grad in zip(self.buckets[handle.bucket_idx].params, unflattened_grads):
                old_param.grad.copy_(new_grad)
        self.handles.clear()


def setup(rank: int, world_size: int, backend: str = "gloo"):
    """
    Setup the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    device_cnt = torch.cuda.device_count()
    if device_cnt == 0:
        raise ValueError("No GPUs available for NCCL backend.")

    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_ddp(rank: int, world_size: int, batch_size:int, num_epochs:int, backend: str = "gloo", flat: bool = False, shard_optimizer: bool = False):
    setup(rank, world_size, backend)
    torch.manual_seed(rank)

    input_dim = 128
    output_dim = 10
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and backend=="nccl" else 'cpu')
    model = ToyModel(input_dim, output_dim).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if shard_optimizer:
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(batch_size, input_dim).to(device)
    y = torch.randint(0, output_dim, (batch_size,)).to(device)
    shard_size = batch_size // world_size
    x_train = x[rank * shard_size:(rank + 1) * shard_size]
    y_train = y[rank * shard_size:(rank + 1) * shard_size]

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()

        if flat:
            params = [param for param in model.parameters() if param.grad is not None]
            grads = [param.grad for param in params]
            flattened_grads = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
            flattened_grads /= world_size

            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
            for old_param, new_grad in zip(params, unflattened_grads):
                old_param.grad = new_grad
        else:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size
        
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    if rank == 0:
        print("Training complete.")
        # Save the model
        torch.save(model.state_dict(), "model.pth")
        print("Model saved.")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes")
    parser.add_argument("--backend", type=str, default="gloo", help="Backend to use (gloo or nccl)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--flat", action="store_true", help="Use flat gradient communication")
    parser.add_argument("--shard_optimizer", action="store_true", help="Use sharded optimizer")
    args = parser.parse_args()

    assert args.world_size > 1, "World size must be greater than 1 for DDP"
    assert args.backend in ["gloo", "nccl"], "Backend must be either 'gloo' or 'nccl'"
    assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.batch_size % args.world_size == 0, "Batch size must be divisible by world size"

    mp.spawn(run_ddp,
        args=(args.world_size, args.batch_size, args.num_epochs, args.backend, args.flat, args.shard_optimizer),
        nprocs=args.world_size,
        join=True)