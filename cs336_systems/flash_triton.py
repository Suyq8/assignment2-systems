import math
from einops import rearrange
import torch
import triton
import triton.language as tl

Q_TILE_SIZE = 32
K_TILE_SIZE = 32

@triton.jit
def flash_fwd(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, 
              q_stride_b, q_stride_n, q_stride_d,
              k_stride_b, k_stride_n, k_stride_d,
              v_stride_b, v_stride_n, v_stride_d,
              o_stride_b, o_stride_n, o_stride_d,
              l_stride_b, l_stride_n,
              Nq: tl.constexpr, Nk: tl.constexpr, head_dim: tl.constexpr,
              q_tile_size: tl.constexpr, k_tile_size: tl.constexpr,
              is_causal: tl.constexpr):
    """
    Flash attention forward kernel.
    Args:
        q_ptr: Pointer to the query tensor.
        k_ptr: Pointer to the key tensor.
        v_ptr: Pointer to the value tensor.
        o_ptr: Pointer to the output tensor.
        l_ptr: Pointer to the log-sum-exp tensor.
        Nq: Number of queries.
        Nk: Number of keys/values.
        head_dim: Dimension of each head.
        q_tile_size: Tile size for queries.
        k_tile_size: Tile size for keys/values.
        is_causal: Whether to apply causal masking.
    """
    batch_idx = tl.program_id(1)
    q_idx = tl.program_id(0)

    q_block_ptr = tl.make_block_ptr(q_ptr+batch_idx*q_stride_b,
                                    shape=(Nq, head_dim),
                                    strides=(q_stride_n, q_stride_d),
                                    offsets=(q_idx*q_tile_size, 0),
                                    block_shape=(q_tile_size, head_dim),
                                    order=(1, 0))
    
    k_block_ptr = tl.make_block_ptr(k_ptr+batch_idx*k_stride_b,
                                    shape=(Nk, head_dim),
                                    strides=(k_stride_n, k_stride_d),
                                    offsets=(0, 0),
                                    block_shape=(k_tile_size, head_dim),
                                    order=(1, 0))
    
    v_block_ptr = tl.make_block_ptr(v_ptr+batch_idx*v_stride_b,
                                    shape=(Nk, head_dim),
                                    strides=(v_stride_n, v_stride_d),
                                    offsets=(0, 0),
                                    block_shape=(k_tile_size, head_dim),
                                    order=(1, 0))
    
    o_block_ptr = tl.make_block_ptr(o_ptr+batch_idx*o_stride_b,
                                    shape=(Nq, head_dim),
                                    strides=(o_stride_n, o_stride_d),
                                    offsets=(q_idx*q_tile_size, 0),
                                    block_shape=(q_tile_size, head_dim),
                                    order=(1, 0))
    
    l_block_ptr = tl.make_block_ptr(l_ptr+batch_idx*l_stride_b,
                                    shape=(Nq,),
                                    strides=(l_stride_n,),
                                    offsets=(q_idx*q_tile_size,),
                                    block_shape=(q_tile_size,),
                                    order=(0,))
    
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_last = tl.zeros((q_tile_size, head_dim), dtype=tl.float32)
    m_last = tl.full((q_tile_size,), float("-inf"), dtype=tl.float32)
    l_last = tl.zeros((q_tile_size,), dtype=tl.float32)

    Tk = (Nk+k_tile_size-1)//k_tile_size
    scale = 1.0 / (head_dim ** 0.5)

    if is_causal:
        q_offset = q_idx*q_tile_size+tl.arange(0, q_tile_size)

    for k_idx in range(Tk):
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s = tl.dot(q, tl.trans(k))*scale

        if is_causal:
            k_offset = k_idx*k_tile_size+tl.arange(0, k_tile_size)
            mask = q_offset[:, None] >= k_offset[None, :]
            s = tl.where(mask, s, -1e6)

        m = tl.maximum(m_last, tl.max(s, axis=1))

        tmp = tl.exp(m_last - m)
        P = tl.exp(s - m[:, None]) # (batch, q_tile_size, k_tile_size)
        l = tmp*l_last+tl.sum(P, axis=-1)
        o = tmp[:, None]*o_last+tl.dot(P.to(v.dtype), v)

        k_block_ptr = k_block_ptr.advance((k_tile_size, 0))
        v_block_ptr = v_block_ptr.advance((k_tile_size, 0))
        o_last = o
        m_last = m
        l_last = l

    tl.store(o_block_ptr, (o_last/l_last[:, None]).to(o_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(l_block_ptr, (m_last+tl.log(l_last)).to(l_block_ptr.type.element_ty), boundary_check=(0,))

@triton.jit
def compute_d(do_ptr, o_ptr, d_ptr,
              do_stride_b, do_stride_n, do_stride_d,
              o_stride_b, o_stride_n, o_stride_d,
              d_stride_b, d_stride_n,
              Nq: tl.constexpr, head_dim: tl.constexpr,
              q_tile_size: tl.constexpr):
    """Compute D = rowsum(dO * O) (batch, ..., seq_len_q)"""
    batch_idx = tl.program_id(1)
    q_idx = tl.program_id(0)

    do_block_ptr = tl.make_block_ptr(do_ptr+batch_idx*do_stride_b,
                                     shape=(Nq, head_dim),
                                     strides=(do_stride_n, do_stride_d),
                                     offsets=(q_idx*q_tile_size, 0),
                                     block_shape=(q_tile_size, head_dim),
                                     order=(1, 0))
    
    o_block_ptr = tl.make_block_ptr(o_ptr+batch_idx*o_stride_b,
                                    shape=(Nq, head_dim),
                                    strides=(o_stride_n, o_stride_d),
                                    offsets=(q_idx*q_tile_size, 0),
                                    block_shape=(q_tile_size, head_dim),
                                    order=(1, 0))
    
    d_block_ptr = tl.make_block_ptr(d_ptr+batch_idx*d_stride_b,
                                    shape=(Nq,),
                                    strides=(d_stride_n,),
                                    offsets=(q_idx*q_tile_size,),
                                    block_shape=(q_tile_size,),
                                    order=(0,))
    
    do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")

    tl.store(d_block_ptr, tl.sum(do*o, axis=-1), boundary_check=(0,))


@triton.jit
def flash_bwd(q_ptr, k_ptr, v_ptr, l_ptr, do_ptr, d_ptr, dq_ptr, dk_ptr, dv_ptr,
              q_stride_b, q_stride_n, q_stride_d,
              k_stride_b, k_stride_n, k_stride_d,
              v_stride_b, v_stride_n, v_stride_d,
              l_stride_b, l_stride_n,
              do_stride_b, do_stride_n, do_stride_d,
              d_stride_b, d_stride_n,
              dq_stride_b, dq_stride_n, dq_stride_d,
              dk_stride_b, dk_stride_n, dk_stride_d,
              dv_stride_b, dv_stride_n, dv_stride_d,
              Nq: tl.constexpr, Nk: tl.constexpr, head_dim: tl.constexpr,
              q_tile_size: tl.constexpr, k_tile_size: tl.constexpr,
              is_causal: tl.constexpr):
    batch_idx = tl.program_id(1)
    k_idx = tl.program_id(0)

    q_block_ptr = tl.make_block_ptr(q_ptr+batch_idx*q_stride_b,
                                    shape=(Nq, head_dim), 
                                    strides=(q_stride_n, q_stride_d), 
                                    offsets=(0, 0), 
                                    block_shape=(q_tile_size, head_dim), 
                                    order=(1, 0))

    k_block_ptr = tl.make_block_ptr(k_ptr+batch_idx*k_stride_b,
                                    shape=(Nk, head_dim), 
                                    strides=(k_stride_n, k_stride_d), 
                                    offsets=(k_idx*k_tile_size, 0), 
                                    block_shape=(k_tile_size, head_dim), 
                                    order=(1, 0))
    
    v_block_ptr = tl.make_block_ptr(v_ptr+batch_idx*v_stride_b,
                                    shape=(Nk, head_dim), 
                                    strides=(v_stride_n, v_stride_d), 
                                    offsets=(k_idx*k_tile_size, 0), 
                                    block_shape=(k_tile_size, head_dim), 
                                    order=(1, 0))
    
    l_block_ptr = tl.make_block_ptr(l_ptr+batch_idx*l_stride_b,
                                    shape=(Nq,), 
                                    strides=(l_stride_n,), 
                                    offsets=(0,), 
                                    block_shape=(q_tile_size,), 
                                    order=(0,))
    
    do_block_ptr = tl.make_block_ptr(do_ptr+batch_idx*do_stride_b,
                                    shape=(Nq, head_dim), 
                                    strides=(do_stride_n, do_stride_d), 
                                    offsets=(0, 0), 
                                    block_shape=(q_tile_size, head_dim), 
                                    order=(1, 0))
    
    d_block_ptr = tl.make_block_ptr(d_ptr+batch_idx*d_stride_b,
                                    shape=(Nq,), 
                                    strides=(d_stride_n,), 
                                    offsets=(0,), 
                                    block_shape=(q_tile_size,), 
                                    order=(0,))
    
    offset_n = tl.arange(0, q_tile_size)
    offset_d = tl.arange(0, head_dim)
    dq_ptrs = dq_ptr+batch_idx*dq_stride_b+offset_n[:, None]*dq_stride_n+offset_d[None, :]*dq_stride_d
    
    dk_block_ptr = tl.make_block_ptr(dk_ptr+batch_idx*dk_stride_b,
                                    shape=(Nk, head_dim), 
                                    strides=(dk_stride_n, dk_stride_d), 
                                    offsets=(k_idx*k_tile_size, 0), 
                                    block_shape=(k_tile_size, head_dim), 
                                    order=(1, 0))
    
    dv_block_ptr = tl.make_block_ptr(dv_ptr+batch_idx*dv_stride_b,
                                    shape=(Nk, head_dim), 
                                    strides=(dv_stride_n, dv_stride_d), 
                                    offsets=(k_idx*k_tile_size, 0), 
                                    block_shape=(k_tile_size, head_dim), 
                                    order=(1, 0))

    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dk = tl.zeros((k_tile_size, head_dim), dtype=tl.float32)
    dv = tl.zeros((k_tile_size, head_dim), dtype=tl.float32)

    Tq = (Nq+q_tile_size-1)//q_tile_size
    scale = 1.0 / (head_dim ** 0.5)

    if is_causal:
        k_offset = k_idx*k_tile_size+tl.arange(0, k_tile_size)

    for q_idx in range(Tq):
        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")
        l = tl.load(l_block_ptr, boundary_check=(0,), padding_option="zero")
        d = tl.load(d_block_ptr, boundary_check=(0,), padding_option="zero")
        
        s = tl.dot(q, tl.trans(k))*scale

        if is_causal:
            q_offset = q_idx*q_tile_size+tl.arange(0, q_tile_size)
            mask = q_offset[:, None] >= k_offset[None, :]
            s = tl.where(mask, s, -1e6)

        p = tl.exp(s-l[:, None])

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = (p*(dp-d[:, None])*scale).to(q.dtype)
        dk += tl.dot(tl.trans(ds), q)

        dq = tl.dot(ds, k)
        tl.atomic_add(dq_ptrs, dq)

        q_block_ptr = q_block_ptr.advance((q_tile_size, 0))
        do_block_ptr = do_block_ptr.advance((q_tile_size, 0))
        l_block_ptr = l_block_ptr.advance((q_tile_size,))
        d_block_ptr = d_block_ptr.advance((q_tile_size,))
        dq_ptrs += q_tile_size*q_stride_n

    tl.store(dk_block_ptr, dk.to(dk_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dv_block_ptr, dv.to(dv_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool=False) -> torch.Tensor:
        """
        Forward pass of the FlashAttention function.

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

        Q = rearrange(Q, '... q d -> (...) q d')
        K = rearrange(K, '... k d -> (...) k d')
        V = rearrange(V, '... k d -> (...) k d')

        Tq = (Q.size(-2)+q_tile_size-1)//q_tile_size

        O = torch.empty_like(Q)
        L = torch.empty((Q.size(0), Nq), device=Q.device, dtype=torch.float32)

        flash_fwd[(Tq, math.prod(B))](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, head_dim,
            q_tile_size, k_tile_size,
            is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O.reshape(*B, Nq, head_dim)
    
    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass of the FlashAttention function.

        Args:
            ctx: The context object that contains information from the forward pass.
            dO: The gradient of the output tensor with respect to the loss.

        Returns:
            Gradients with respect to Q, K, and V.
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        B = dO.shape[:-2]
        Nq, head_dim = Q.size(-2), Q.size(-1)
        Nk = K.size(-2)

        q_tile_size = min(Q_TILE_SIZE, Nq)
        k_tile_size = min(K_TILE_SIZE, Nk)

        dO = rearrange(dO, '... q d -> (...) q d')

        Tq = (Nq+q_tile_size-1)//q_tile_size

        D = torch.empty_like(L, device=Q.device, dtype=torch.float32)
        compute_d[(Tq, math.prod(B))](
            dO, O, D,
            dO.stride(0), dO.stride(1), dO.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            D.stride(0), D.stride(1),
            Nq, head_dim,
            q_tile_size
        )

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        Tk = (Nk+k_tile_size-1)//k_tile_size

        flash_bwd[(Tk, math.prod(B))](
            Q, K, V, L, dO, D, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk, head_dim,
            q_tile_size, k_tile_size,
            is_causal
        )

        return dQ.reshape(*B, Nq, head_dim), dK.reshape(*B, Nk, head_dim), dV.reshape(*B, Nk, head_dim), None