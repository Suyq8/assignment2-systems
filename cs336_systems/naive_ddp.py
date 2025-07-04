import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class ToyModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear(x)))

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

def run_ddp(rank: int, world_size: int, batch_size:int, num_epochs:int, backend: str = "gloo"):
    setup(rank, world_size, backend)
    torch.manual_seed(rank)

    input_dim = 128
    output_dim = 10
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() and backend=="nccl" else 'cpu')
    model = ToyModel(input_dim, output_dim).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
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
    args = parser.parse_args()

    assert args.world_size > 1, "World size must be greater than 1 for DDP"
    assert args.backend in ["gloo", "nccl"], "Backend must be either 'gloo' or 'nccl'"
    assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.batch_size % args.world_size == 0, "Batch size must be divisible by world size"

    mp.spawn(run_ddp,
        args=(args.world_size, args.batch_size, args.num_epochs, args.backend),
        nprocs=args.world_size,
        join=True)