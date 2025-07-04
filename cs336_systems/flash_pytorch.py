from einops import einsum
import torch

Q_TILE_SIZE = 128
K_TILE_SIZE = 128


class FlashPytorch(torch.autograd.Function):
    """
    FlashPytorch is a custom autograd function that performs a forward and backward pass for a neural network.
    It uses the PyTorch library to define the forward and backward methods, which are called during the training process.
    """

    @staticmethod
    def forward(
        ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the FlashPytorch function.

        Args:
            ctx: The context object that can be used to store information for the backward pass.
            Q: The query tensor. (batch, ..., seq_len_q, head_dim)
            K: The key tensor. (batch, ..., seq_len_k, head_dim)
            V: The value tensor. (batch, ..., seq_len_k, head_dim)
            is_causal: causal masking

        Returns:
            The output tensor after applying flash attention.
        """
        B = Q.shape[:-2]
        Nq, head_dim = Q.size(-2), Q.size(-1)
        Nk = K.size(-2)

        q_tile_size = min(Q_TILE_SIZE, Nq)
        k_tile_size = min(K_TILE_SIZE, Nk)

        Q = Q.reshape(*B, Nq // q_tile_size, q_tile_size, -1)
        K = K.reshape(*B, Nk // k_tile_size, k_tile_size, -1)
        V = V.reshape(*B, Nk // k_tile_size, k_tile_size, -1)

        Tq = Q.size(-3)
        Tk = K.size(-3)

        O = torch.empty((*B, Nq, head_dim), device=Q.device, dtype=Q.dtype).view(
            *B, Tq, q_tile_size, head_dim
        )  # output
        L = torch.empty((*B, Nq), device=Q.device, dtype=Q.dtype).view(
            *B, Tq, q_tile_size
        )  # log_sum_exp
        scale = 1.0 / (head_dim**0.5)

        for i in range(Tq):
            q = Q[..., i, :, :]  # (batch, ..., q_tile_size, head_dim)
            o_last = torch.full(
                (*B, q_tile_size, head_dim), 0.0, device=Q.device, dtype=Q.dtype
            )  # (batch, ..., q_tile_size, head_dim)
            m_last = torch.full(
                (*B, q_tile_size, 1), -torch.inf, device=Q.device, dtype=Q.dtype
            )  # (batch, ..., q_tile_sizem, 1)
            l_last = torch.full(
                (*B, q_tile_size, 1), 0.0, device=Q.device, dtype=Q.dtype
            )  # (batch, ..., q_tile_size, 1)
            for j in range(Tk):
                k = K[..., j, :, :]  # (batch, ..., k_tile_size, head_dim)
                v = V[..., j, :, :]  # (batch, ..., k_tile_size, head_dim)

                # Compute tile of pre-softmax attention scores
                s = (
                    einsum(q, k, "... q h, ... k h -> ... q k") * scale
                )  # (batch, q_tile_size, k_tile_size)

                # running maximum
                m = torch.max(m_last, torch.max(s, dim=-1, keepdim=True).values)

                tmp = torch.exp(m_last - m)
                P = torch.exp(s - m)  # (batch, q_tile_size, k_tile_size)
                l = tmp * l_last + torch.sum(P, dim=-1, keepdim=True)
                o = torch.diag_embed(tmp.squeeze_(dim=-1)) @ o_last + P @ v

                o_last = o
                m_last = m
                l_last = l

            O[:, i] = torch.diag_embed(1.0 / l_last.squeeze_(dim=-1)) @ o_last
            L[:, i] = m_last.squeeze_(dim=-1) + torch.log(l_last.squeeze_(dim=-1))

        O = O.view(*B, Tq * q_tile_size, head_dim)
        L = L.view(*B, Tq * q_tile_size)
        Q = Q.reshape(*B, Nq, head_dim)
        K = K.reshape(*B, Nk, head_dim)
        V = V.reshape(*B, Nk, head_dim)

        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the FlashPytorch function.

        Args:
            ctx: The context object that contains information from the forward pass.
            dO: The gradient of the output tensor with respect to the loss.

        Returns:
            Gradients of the input tensors Q, K, and V.
        """
        Q, K, V, O, L = ctx.saved_tensors

        head_dim = Q.size(-1)

        scale = 1.0 / (head_dim**0.5)

        S = einsum(Q, K, "... q h, ... k h -> ... q k") * scale  # (batch, ..., Nq, Nk)
        P = torch.exp(S - L.unsqueeze(-1))

        dV = einsum(P, dO, "... q k, ... q h -> ... k h")
        dP = einsum(dO, V, "... q h, ... k h -> ... q k")
        dS = P * (dP - torch.sum(O * dO, dim=-1, keepdim=True))
        dQ = dS @ K * scale
        dK = einsum(dS, Q, "... q k, ... q h -> ... k h") * scale

        return dQ, dK, dV
