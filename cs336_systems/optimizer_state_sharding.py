from typing import Type
import torch
import torch.distributed as dist

class ShardedOptimizer(torch.optim.Optimizer):
    """
    A wrapper around PyTorch optimizers to support sharding across multiple devices.
    """

    def __init__(
        self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: any
    ):
        """
        Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter
        groups, in case the user wants to use different hyperparameters, such as learning rates, for differ-
        ent parts of the model); these parameters will be sharded across all the ranks. The optimizer_cls
        parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW). Finally, any remain-
        ing keyword arguments are forwarded to the constructor of the optimizer_cls. Make sure to
        call the torch.optim.Optimizer super-class constructor in this method.
        """
        self.optimizer_cls = optimizer_cls
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.param_rank = {}
        self.optimizer = None
        self.kwargs = kwargs

        super().__init__(params, defaults={})

    def step(self, closure=None, **kwargs):
        """
        Calls the wrapped optimizer's step() method with the provided closure and keyword arguments. After
        updating the parameters, synchronize with the other ranks.
        """
        if self.optimizer is not None:
            self.optimizer.step(closure, **kwargs)

        for p, rank in self.param_rank.items():
            dist.broadcast(p.data, src=rank)

    def add_param_group(self, param_group: dict[str, any]):
        """
        This method should add a parameter group to the sharded optimizer. This is called during construction 
        of the sharded optimizer by the super-class constructor and may also be called during training 
        (e.g., for gradually unfreezing layers in a model). As a result, this method should handle assigning 
        the model's parameters among the ranks.
        """
        local_shard = []
        for i, param in enumerate(param_group["params"]):
            self.param_rank[param] = i%self.world_size
            if self.param_rank[param] == self.rank:
                local_shard.append(param)

        new_param_group = {k: v for k, v in param_group.items() if k != "params"}
        new_param_group["params"] = local_shard

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(local_shard, **self.kwargs)
        else:
            self.optimizer.add_param_group(new_param_group)

        super().add_param_group(param_group)