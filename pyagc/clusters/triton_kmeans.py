# Adapted from https://github.com/svg-project/flash-kmeans

from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _has_triton = True
except ImportError:
    _has_triton = False
    triton = None
    tl = None


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ============================================================
# All Triton kernels and related helpers are only defined
# when Triton is available.
# ============================================================
if _has_triton:
    import tqdm

    _TUNE_CONFIGS = [
        triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for wp in [4, 8]
        for num_stages in [1, 2, 4]
    ]


    def _cfg_keep(conf):
        """Basic heuristic to prune unbalanced configs."""
        BN = conf.kwargs["BLOCK_N"]
        BK = conf.kwargs["BLOCK_K"]
        if BN * BK < 32 * 32 and conf.num_warps > 4:
            return False
        return True


    _TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


    @triton.autotune(_TUNE_CONFIGS, key=["N", "K"])
    @triton.jit
    def _euclid_assign_kernel(
            x_ptr,  # *f16 / *f32 [N, D]
            c_ptr,  # *f16 / *f32 [K, D]
            x_sq_ptr,  # *f32         [N]
            c_sq_ptr,  # *f32         [K]
            out_ptr,  # *i32         [N]
            N: tl.constexpr,
            K: tl.constexpr,
            D: tl.constexpr,
            stride_x_n: tl.constexpr,
            stride_x_d: tl.constexpr,
            stride_c_k: tl.constexpr,
            stride_c_d: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
    ):
        """Each program handles a tile of BLOCK_N points."""
        pid_n = tl.program_id(0)

        n_start = pid_n * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Load x tile  (BLOCK_N, D)
        offs_d = tl.arange(0, D)
        x_ptrs = x_ptr + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
        x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
        x_tile = x_tile.to(tl.float32)

        # Pre-load x_sq for the tile  (BLOCK_N,)
        xsq_ptrs = x_sq_ptr + n_offsets
        x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        # Init best distance / index
        best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
        best_idx = tl.zeros((BLOCK_N,), tl.int32)

        # Iterate over centroids in chunks of BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            # Load centroid tile  (D, BLOCK_K)
            c_ptrs = c_ptr + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
            c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
            c_tile = c_tile.to(tl.float32)

            # load c_sq for the tile  (BLOCK_K,)
            csq_ptrs = c_sq_ptr + k_offsets
            cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
            cross = tl.dot(x_tile, c_tile).to(tl.float32)

            # Squared Euclidean distance
            dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
            dist = tl.maximum(dist, 0.0)

            # Mask out invalid centroid columns
            dist = tl.where(k_mask[None, :], dist, 3.4e38)

            curr_min = tl.min(dist, axis=1)
            curr_idx = tl.argmin(dist, axis=1)

            update = curr_min < best_dist
            best_dist = tl.where(update, curr_min, best_dist)
            best_idx = tl.where(update, k_start + curr_idx, best_idx)

        # Write results
        out_ptrs = out_ptr + n_offsets
        tl.store(out_ptrs, best_idx, mask=n_mask)


    @triton.autotune(_TUNE_CONFIGS, key=["N", "K"])
    @triton.jit
    def _cosine_assign_kernel(
            x_ptr,  # *f16 / *f32 [N, D]
            c_ptr,  # *f16 / *f32 [K, D]
            out_ptr,  # *i32         [N]
            N: tl.constexpr,
            K: tl.constexpr,
            D: tl.constexpr,
            stride_x_n: tl.constexpr,
            stride_x_d: tl.constexpr,
            stride_c_k: tl.constexpr,
            stride_c_d: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
    ):
        """Each program handles a tile of BLOCK_N points for cosine similarity."""
        pid_n = tl.program_id(0)

        n_start = pid_n * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Load x tile  (BLOCK_N, D)
        offs_d = tl.arange(0, D)
        x_ptrs = x_ptr + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
        x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
        x_tile = x_tile.to(tl.float32)

        # Init best distance / index
        best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)
        best_idx = tl.zeros((BLOCK_N,), tl.int32)

        # Iterate over centroids in chunks of BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            # Load centroid tile  (D, BLOCK_K)
            c_ptrs = c_ptr + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
            c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
            c_tile = c_tile.to(tl.float32)

            # Compute cosine distance (BLOCK_N, BLOCK_K) = x_tile @ c_tile
            cross = tl.dot(x_tile, c_tile).to(tl.float32)

            # Mask out invalid centroid columns
            dist = tl.where(k_mask[None, :], cross, 0.0)

            curr_max = tl.max(dist, axis=1)
            curr_idx = tl.argmax(dist, axis=1)

            update = curr_max > best_dist
            best_dist = tl.where(update, curr_max, best_dist)
            best_idx = tl.where(update, k_start + curr_idx, best_idx)

        # Write results
        out_ptrs = out_ptr + n_offsets
        tl.store(out_ptrs, best_idx, mask=n_mask)


    def euclid_assign_triton(x: torch.Tensor, centroids: torch.Tensor, x_sq: torch.Tensor = None,
                             out: torch.Tensor = None, c_sq: torch.Tensor = None) -> torch.Tensor:
        """Return nearest-centroid indices using Triton kernel.

        Args:
            x         : (N, D) float16 / float32 (on CUDA)
            centroids : (K, D) same dtype/device as x
            x_sq      : (N,)   float32 – ||x||^2 per point (optional)
            out       : (N,)   int32   – pre-allocated output tensor (optional)
            c_sq      : (K,)   float32 – ||centroids||^2 per centroid (optional)

        Returns:
            cluster_ids (N,) int32
        """
        assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
        assert centroids.dtype == x.dtype, "centroids dtype mismatch"
        assert x.ndim == 2 and centroids.ndim == 2, "Expected 2D tensors"

        N, D = x.shape
        K, D2 = centroids.shape
        assert D == D2, "Feature dimension mismatch"

        if x_sq is None:
            x_sq = (x.to(torch.float32) ** 2).sum(-1)
        assert x_sq.shape == (N,), "x_sq shape mismatch"

        if out is None:
            out = torch.empty(N, device=x.device, dtype=torch.int32)

        if c_sq is None:
            c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

        stride_x_n, stride_x_d = x.stride()
        stride_c_k, stride_c_d = centroids.stride()

        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

        _euclid_assign_kernel[grid](
            x, centroids, x_sq, c_sq, out,
            N, K, D,
            stride_x_n, stride_x_d,
            stride_c_k, stride_c_d,
        )
        return out


    def cosine_assign_triton(x: torch.Tensor, centroids: torch.Tensor,
                             out: torch.Tensor = None) -> torch.Tensor:
        """Return nearest(cosine similarity)-centroid indices using Triton kernel.

        Args:
            x         : (N, D) float16 / float32 (on CUDA)
            centroids : (K, D) same dtype/device as x
            out       : (N,)   int32   – pre-allocated output tensor (optional)

        Returns:
            cluster_ids (N,) int32
        """
        assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
        assert centroids.dtype == x.dtype, "centroids dtype mismatch"
        assert x.ndim == 2 and centroids.ndim == 2, "Expected 2D tensors"

        N, D = x.shape
        K, D2 = centroids.shape
        assert D == D2, "Feature dimension mismatch"

        if out is None:
            out = torch.empty(N, device=x.device, dtype=torch.int32)

        stride_x_n, stride_x_d = x.stride()
        stride_c_k, stride_c_d = centroids.stride()

        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

        _cosine_assign_kernel[grid](
            x, centroids, out,
            N, K, D,
            stride_x_n, stride_x_d,
            stride_c_k, stride_c_d,
        )
        return out


    @triton.jit
    def _centroid_update_kernel(
            x_ptr,  # *f16 / *f32 [N, D]
            cluster_ptr,  # *i32        [N]
            sum_ptr,  # *f32        [K, D]
            count_ptr,  # *i32        [K]
            # --- strides (elements) ---
            stride_x_n, stride_x_d,
            stride_sum_k, stride_sum_d,
            N: tl.constexpr,
            D: tl.constexpr,
            K: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        """Each program processes 1 token across BLOCK_D dims using atomics."""
        pid = tl.program_id(axis=0)
        token_idx = pid

        if token_idx >= N:
            return

        # pointer to this token's feature vector
        x_offset = token_idx * stride_x_n
        x_tok_ptr = x_ptr + x_offset

        cluster_idx = tl.load(cluster_ptr + token_idx)
        cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)

        # base ptr for centroid accum array
        centroid_base = cluster_idx * stride_sum_k

        offs = tl.arange(0, BLOCK_D)
        for d_start in range(0, D, BLOCK_D):
            mask = offs + d_start < D
            feats = tl.load(x_tok_ptr + (d_start + offs) * stride_x_d, mask=mask, other=0.0)
            feats = feats.to(tl.float32)
            dest_ptr = sum_ptr + centroid_base + (d_start + offs) * stride_sum_d
            tl.atomic_add(dest_ptr, feats, mask=mask)

        tl.atomic_add(count_ptr + cluster_idx, 1)


    @triton.jit
    def _centroid_update_chunk_kernel(
            x_ptr,  # *f16 / *f32 [N, D] – ORIGINAL ORDER
            sorted_idx_ptr,  # *i32        [N]    – indices after sort
            sorted_cluster_ptr,  # *i32        [N]    – cluster ids in sorted order
            sum_ptr,  # *f32        [K, D]
            count_ptr,  # *i32        [K]
            # strides
            stride_x_n, stride_x_d,
            N: tl.constexpr,
            D: tl.constexpr,
            K: tl.constexpr,
            BLOCK_N: tl.constexpr,
    ):
        """Each program processes BLOCK_N consecutive, already-sorted tokens."""
        pid_chunk = tl.program_id(axis=0)
        chunk_start = pid_chunk * BLOCK_N

        if chunk_start >= N:
            return

        # helper aranges
        offs_token = tl.arange(0, BLOCK_N)
        offs_dim = tl.arange(0, D)

        # token indices & validity mask
        token_idx = chunk_start + offs_token
        valid_tok = token_idx < N
        first_token_idx = chunk_start
        last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

        # Load cluster ids
        first_id = tl.load(sorted_cluster_ptr + first_token_idx)
        last_id = tl.load(sorted_cluster_ptr + last_token_idx)
        all_ids = tl.load(sorted_cluster_ptr + token_idx, mask=valid_tok, other=-1)

        # Load original indices
        all_tokens_idxs = tl.load(sorted_idx_ptr + token_idx, mask=valid_tok, other=-1)

        for cid in range(first_id, last_id + 1):
            cluster_mask = all_ids == cid
            cluster_size = tl.sum(cluster_mask.to(tl.int32))
            if cluster_size != 0:
                row_ptrs = x_ptr + all_tokens_idxs[:, None] * stride_x_n + offs_dim[None, :] * stride_x_d
                cluster_feats = tl.load(row_ptrs, mask=cluster_mask[:, None], other=0.0)
                cluster_feats = cluster_feats.to(tl.float32)
                sum_feats = tl.sum(cluster_feats, axis=0)
                dest_ptr = sum_ptr + cid * D + offs_dim
                tl.atomic_add(dest_ptr, sum_feats)
                tl.atomic_add(count_ptr + cid, cluster_size)


    def triton_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor,
                                      old_centroids: torch.Tensor):
        """Compute centroids using custom Triton kernel.

        Args:
            x_norm (Tensor): (N, D) normalized input vectors
            cluster_ids (LongTensor): (N,) cluster assignment per point
            old_centroids (Tensor): (K, D) previous centroids

        Returns:
            Tensor: (K, D) updated and L2-normalized centroids
        """
        assert x_norm.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
        assert x_norm.ndim == 2 and old_centroids.ndim == 2, "Expected 2D tensors"

        N, D = x_norm.shape
        K, D2 = old_centroids.shape
        assert D == D2, "Feature dimension mismatch"
        assert cluster_ids.shape == (N,)

        # Allocate accumulation buffers
        centroid_sums = torch.zeros((K, D), device=x_norm.device, dtype=torch.float32)
        centroid_counts = torch.zeros(K, device=x_norm.device, dtype=torch.int32)

        BLOCK_D = 128
        grid = (N,)
        _centroid_update_kernel[grid](
            x_norm,
            cluster_ids.to(torch.int32),
            centroid_sums,
            centroid_counts,
            x_norm.stride(0), x_norm.stride(1),
            centroid_sums.stride(0), centroid_sums.stride(1),
            N, D, K,
            BLOCK_D=BLOCK_D,
        )

        # Compute means; keep old centroid if empty cluster
        counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f

        # For clusters with zero count, revert to old centroids
        zero_mask = (centroid_counts == 0).unsqueeze(-1)
        centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

        centroids = centroids.to(x_norm.dtype)
        centroids = F.normalize(centroids, p=2, dim=-1)
        return centroids


    def triton_centroid_update_euclid(x: torch.Tensor, cluster_ids: torch.Tensor,
                                      old_centroids: torch.Tensor):
        """Compute centroids for Euclidean KMeans using Triton.

        Args:
            x (Tensor): (N, D) input vectors
            cluster_ids (LongTensor): (N,) cluster assignment per point
            old_centroids (Tensor): (K, D) previous centroids

        Returns:
            Tensor: (K, D) updated centroids
        """
        assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
        assert x.ndim == 2 and old_centroids.ndim == 2, "Expected 2D tensors"

        N, D = x.shape
        K, D2 = old_centroids.shape
        assert D == D2, "Feature dimension mismatch"
        assert cluster_ids.shape == (N,)

        # Allocate accumulation buffers
        centroid_sums = torch.zeros((K, D), device=x.device, dtype=torch.float32)
        centroid_counts = torch.zeros(K, device=x.device, dtype=torch.int32)

        BLOCK_D = 128
        grid = (N,)

        _centroid_update_kernel[grid](
            x,
            cluster_ids.to(torch.int32),
            centroid_sums,
            centroid_counts,
            x.stride(0), x.stride(1),
            centroid_sums.stride(0), centroid_sums.stride(1),
            N, D, K,
            BLOCK_D=BLOCK_D,
        )

        # Compute means; keep old centroid if empty cluster
        counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f

        # For clusters with zero count, revert to old centroids
        zero_mask = (centroid_counts == 0).unsqueeze(-1)
        centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

        return centroids.to(x.dtype)


    def triton_centroid_update_sorted_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor,
                                             old_centroids: torch.Tensor, *, BLOCK_N: int = 256):
        """Fast centroid update assuming cluster_ids are sorted along N.

        Args:
            x_norm (Tensor): (N, D) normalized input vectors
            cluster_ids (LongTensor): (N,) cluster assignment per point
            old_centroids (Tensor): (K, D) previous centroids
            BLOCK_N (int): Tokens per Triton program

        Returns:
            Tensor: (K, D) updated and L2-normalized centroids
        """
        assert x_norm.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA"
        assert x_norm.ndim == 2 and old_centroids.ndim == 2, "Expected 2D tensors"

        N, D = x_norm.shape
        K, D2 = old_centroids.shape
        assert D == D2, "Feature dimension mismatch"
        assert cluster_ids.shape == (N,)

        # Sort per-batch
        sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids)
        sorted_idx_int = sorted_idx.to(torch.int32)

        # accumulation buffers
        centroid_sums = torch.zeros((K, D), device=x_norm.device, dtype=torch.float32)
        centroid_cnts = torch.zeros(K, device=x_norm.device, dtype=torch.int32)

        grid = (triton.cdiv(N, BLOCK_N),)
        _centroid_update_chunk_kernel[grid](
            x_norm,
            sorted_idx_int,
            sorted_cluster_ids.to(torch.int32),
            centroid_sums,
            centroid_cnts,
            x_norm.stride(0), x_norm.stride(1),
            N, D, K,
            BLOCK_N=BLOCK_N,
        )

        # finalise
        counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f
        empty_mask = (centroid_cnts == 0).unsqueeze(-1)
        centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
        centroids = centroids.to(x_norm.dtype)
        centroids = F.normalize(centroids, p=2, dim=-1)
        return centroids


    def triton_centroid_update_sorted_euclid(x: torch.Tensor, cluster_ids: torch.Tensor,
                                             old_centroids: torch.Tensor, *, BLOCK_N: int = 256,
                                             centroid_sums: torch.Tensor = None,
                                             centroid_cnts: torch.Tensor = None,
                                             calculate_new: bool = True):
        """Fast centroid update for Euclidean KMeans assuming cluster IDs are pre-sorted.

        Args:
            x (Tensor): (N, D) input feature vectors
            cluster_ids (LongTensor): (N,) cluster assignment
            old_centroids (Tensor): (K, D) previous centroids
            BLOCK_N (int): Tokens per Triton program
            centroid_sums (Tensor): (K, D) pre-allocated accumulation buffer (optional)
            centroid_cnts (Tensor): (K,) pre-allocated count buffer (optional)
            calculate_new (bool): Whether to compute and return new centroids

        Returns:
            Tensor: (K, D) updated centroids or None if calculate_new=False
        """
        assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
        assert x.ndim == 2 and old_centroids.ndim == 2, "Expected 2D tensors"

        N, D = x.shape
        K, D2 = old_centroids.shape
        assert D == D2, "Feature dimension mismatch"

        # Sort cluster assignments
        sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids)
        sorted_idx_int = sorted_idx.to(torch.int32)

        if centroid_sums is None:
            centroid_sums = torch.zeros((K, D), device=x.device, dtype=torch.float32)
        else:
            assert centroid_sums.shape == (K, D)

        if centroid_cnts is None:
            centroid_cnts = torch.zeros(K, device=x.device, dtype=torch.int32)
        else:
            assert centroid_cnts.shape == (K,)

        grid = (triton.cdiv(N, BLOCK_N),)
        _centroid_update_chunk_kernel[grid](
            x,
            sorted_idx_int,
            sorted_cluster_ids.to(torch.int32),
            centroid_sums,
            centroid_cnts,
            x.stride(0), x.stride(1),
            N, D, K,
            BLOCK_N=BLOCK_N,
        )

        if calculate_new:
            counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
            centroids = centroid_sums / counts_f
            empty_mask = (centroid_cnts == 0).unsqueeze(-1)
            centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
            return centroids.to(x.dtype)
        else:
            return None


    # -------------------- Single-iteration kernels --------------------

    def _euclid_iter(x, x_sq, centroids):
        cluster_ids = euclid_assign_triton(x, centroids, x_sq)
        centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)
        shift = (centroids_new - centroids).norm(dim=-1).max()
        return centroids_new, shift, cluster_ids


    def _cosine_iter(x_norm, centroids):
        cluster_ids = cosine_assign_triton(x_norm, centroids)
        centroids_new = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
        shift = (centroids_new - centroids).norm(dim=-1).max()
        return centroids_new, shift, cluster_ids


    def _dot_iter(x, centroids):
        cluster_ids = cosine_assign_triton(x, centroids)
        centroids_new = triton_centroid_update_sorted_cosine(x, cluster_ids, centroids)
        shift = (centroids_new - centroids).norm(dim=-1).max()
        return centroids_new, shift, cluster_ids


    COMPILE_FLAG = False

    try:
        if COMPILE_FLAG:
            _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
            _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
            _dot_iter_compiled = torch.compile(_dot_iter, dynamic=True, mode="reduce-overhead")
        else:
            _euclid_iter_compiled = _euclid_iter
            _cosine_iter_compiled = _cosine_iter
            _dot_iter_compiled = _dot_iter
    except Exception:
        _euclid_iter_compiled = _euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled = _dot_iter


    def kmeans_Euclid(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        """
        KMeans clustering in PyTorch using Euclidean distance.

        Args:
            x: Tensor of shape (N, D), N points, D dims.
            n_clusters: Number of clusters.
            max_iters: Max number of iterations.
            tol: Tolerance for center movement.
            init_centroids: Initial centroids (K, D) or None
            verbose: Print progress.

        Returns:
            cluster_ids: (N,) LongTensor, cluster assignment for each point.
            centroids: (K, D) final cluster centers.
            n_iters: Number of iterations performed.
        """
        assert x.ndim == 2, "x must be 2D tensor (N, D)"
        N, D = x.shape

        # Pre-compute squared L2 norm of all points
        x_sq = (x ** 2).sum(dim=-1)  # (N,)

        if init_centroids is None:
            # Randomly select initial centers from x
            indices = torch.randint(0, N, (n_clusters,), device=x.device)
            centroids = x[indices]  # (K, D)
        else:
            centroids = init_centroids
            assert centroids.shape == (n_clusters, D), "init_centroids shape mismatch"

        for it in range(max_iters):
            centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(x, x_sq, centroids)

            if verbose:
                print(f"Iter {it}, center shift: {center_shift.item():.6f}")

            if center_shift < tol:
                break

            centroids = centroids_new

        return cluster_ids, centroids, it + 1


    def kmeans_Cosine(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        """
        KMeans clustering in PyTorch using Cosine similarity.

        Args:
            x: Tensor of shape (N, D), N points, D dims.
            n_clusters: Number of clusters.
            max_iters: Max number of iterations.
            tol: Tolerance for center movement.
            init_centroids: Initial centroids (K, D) or None
            verbose: Print progress.

        Returns:
            cluster_ids: (N,) LongTensor, cluster assignment for each point.
            centroids: (K, D) final cluster centers.
            n_iters: Number of iterations performed.
        """
        assert x.ndim == 2, "x must be 2D tensor (N, D)"
        N, D = x.shape

        # Normalize input vectors for cosine similarity
        x_norm = F.normalize(x, p=2, dim=-1)  # (N, D)

        if init_centroids is None:
            # Randomly select initial centers from x_norm
            indices = torch.randint(0, N, (n_clusters,), device=x.device)
            centroids = x_norm[indices]  # (K, D)
        else:
            centroids = init_centroids
            assert centroids.shape == (n_clusters, D), "init_centroids shape mismatch"

        centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

        for it in range(max_iters):
            centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

            if verbose:
                print(f"Iter {it}, center shift: {center_shift.item():.6f}")

            if center_shift < tol:
                break

            centroids = centroids_new

        return cluster_ids, centroids, it + 1


    def kmeans_Dot(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        """
        KMeans clustering in PyTorch using raw dot-product as similarity.

        Args:
            x: Tensor of shape (N, D), N points, D dims.
            n_clusters: Number of clusters.
            max_iters: Max number of iterations.
            tol: Tolerance for center movement.
            init_centroids: Initial centroids (K, D) or None
            verbose: Print progress.

        Returns:
            cluster_ids: (N,) LongTensor, cluster assignment for each point.
            centroids: (K, D) final cluster centers.
            n_iters: Number of iterations performed.
        """
        assert x.ndim == 2, "x must be 2D tensor (N, D)"
        N, D = x.shape

        if init_centroids is None:
            indices = torch.randint(0, N, (n_clusters,), device=x.device)
            centroids = x[indices]
        else:
            centroids = init_centroids
            assert centroids.shape == (n_clusters, D), "init_centroids shape mismatch"

        for it in range(max_iters):
            centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

            if verbose:
                print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")

            if center_shift < tol:
                break

            centroids = centroids_new

        return cluster_ids, centroids, it + 1


    def _require_cuda():
        """Check if CUDA is available."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run the Triton-backed k-means implementation.")


    class TritonKMeans:
        """
        Fast K-Means clustering implemented with Triton GPU kernels.

        This implementation provides an interface compatible with TorchKMeans
        while leveraging Triton kernels for improved performance.

        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric to use. Options: 'euclidean', 'cosine', 'dot'
        init : str or torch.Tensor, default='k-means++'
            Method for initialization: 'k-means++', 'random' or user-specified
            tensor of shape (n_clusters, n_features).
        random_state : int, optional
            Random seed for centroid initialization.
        n_clusters : int, default=8
            Number of clusters (k).
        n_init : int, default=10
            Number of times the algorithm will be run with different centroid seeds.
            The final result will be the best output of n_init consecutive runs.
        max_iter : int, default=300
            Maximum number of iterations for a single run.
        tol : float, default=1e-4
            Relative tolerance with regards to inertia to declare convergence.
        verbose : bool, default=False
            Whether to print per-iteration info.
        dtype : torch.dtype, optional
            Compute data type for algorithm.
        device : torch.device, optional
            Target device. Defaults to "cuda:0" when available.
            Currently, only CUDA devices are supported.
        distributed : bool, default=False
            Reserved for future distributed training support (currently not implemented).
        """

        def __init__(
                self,
                metric: str = 'euclidean',
                init: Union[str, torch.Tensor] = 'k-means++',
                random_state: Optional[int] = None,
                n_clusters: int = 8,
                n_init: int = 10,
                max_iter: int = 300,
                tol: float = 1e-4,
                verbose: bool = False,
                dtype: Optional[torch.dtype] = None,
                device: Optional[torch.device] = None,
                distributed: bool = False,
        ):
            _require_cuda()

            self.metric = metric.lower()
            if self.metric not in ['euclidean', 'cosine', 'dot']:
                raise ValueError(
                    f'Invalid metric value. Must be either "euclidean", "cosine" or "dot". '
                    f'But got "{metric}".'
                )

            # Set distance function based on metric
            from pyagc.clusters.torch_kmeans import _pairwise_euclidean, _pairwise_cosine, _pairwise_dot
            self.distance_metric = {
                'euclidean': _pairwise_euclidean,
                'cosine': _pairwise_cosine,
                'dot': _pairwise_dot
            }[self.metric]

            self.n_clusters = int(n_clusters)
            self.n_init = int(n_init)
            self.max_iter = int(max_iter)
            self.tol = float(tol)
            self.verbose = bool(verbose)
            self.dtype = dtype
            self.init = init

            if isinstance(self.init, torch.Tensor):
                self.n_init = 1

            if random_state is None:
                random_state = 0
            self.random_state = int(random_state)

            # Device setup
            if device is None:
                self.device = torch.device("cuda:0")
            else:
                if device.type != "cuda":
                    raise ValueError("Only CUDA devices are supported.")
                self.device = device

            # Model state
            self.cluster_centers_: Optional[torch.Tensor] = None
            self.labels_: Optional[torch.Tensor] = None
            self.inertia_: Optional[float] = None

            # Statistics from all runs
            self.stats = {'state': [], 'inertia': [], 'label': []}

            # Distributed training (reserved for future use)
            self.distributed = distributed
            if self.distributed:
                raise NotImplementedError("Distributed training is not yet supported for TritonKMeans.")

            # Backward compatibility attributes
            self.d = None  # Will be set during fit
            self.k = self.n_clusters
            self.niter = self.max_iter
            self.seed = self.random_state
            self.centroids = None  # Alias for cluster_centers_
            self.cluster_ids = None  # Alias for labels_

        @torch.no_grad()
        def initialize(self, X: torch.Tensor, random_state: int) -> torch.Tensor:
            """
            Initializes the cluster centers.

            Parameters
            ----------
            X : torch.Tensor
                The input data of shape (n_samples, n_features).
            random_state : int
                The random seed.

            Returns
            -------
            torch.Tensor
                Initialized cluster centers of shape (n_clusters, n_features).
            """
            num_samples = X.size(0)

            if isinstance(self.init, str):
                generator = torch.Generator(device=str(X.device)).manual_seed(random_state)

                if self.init == 'random':
                    indices = torch.randperm(num_samples, generator=generator, device=X.device)[:self.n_clusters]
                    init_state = X[indices].clone()
                elif self.init == 'k-means++':
                    from pyagc.clusters.torch_kmeans import _kmeans_plusplus
                    init_state, _ = _kmeans_plusplus(
                        X,
                        n_clusters=self.n_clusters,
                        random_state=random_state,
                        pairwise_distance=self.distance_metric
                    )
                else:
                    raise NotImplementedError(f"Unknown init method: {self.init}")
            elif isinstance(self.init, torch.Tensor):
                init_state = self.init.to(device=X.device, dtype=X.dtype)
                assert init_state.shape == (self.n_clusters, X.shape[1]), \
                    f"init shape mismatch: expected ({self.n_clusters}, {X.shape[1]}), got {init_state.shape}"
            else:
                raise NotImplementedError(f"Unsupported init type: {type(self.init)}")

            return init_state

        @torch.no_grad()
        def _single_iteration(self, x: torch.Tensor, centroids: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
            """
            Performs a single k-means iteration: assignment + update.

            Parameters
            ----------
            x : torch.Tensor
                Input data of shape (N, D).
            centroids : torch.Tensor
                Current centroids of shape (K, D).

            Returns
            -------
            Tuple[torch.Tensor, float, torch.Tensor]
                (new_centroids, center_shift, cluster_ids)
            """
            if self.metric == 'euclidean':
                x_sq = (x ** 2).sum(dim=-1)
                new_centroids, center_shift, cluster_ids = _euclid_iter_compiled(x, x_sq, centroids)
            elif self.metric == 'cosine':
                # x is l2-normalized
                centroids = F.normalize(centroids, p=2, dim=-1)
                new_centroids, center_shift, cluster_ids = _cosine_iter_compiled(x, centroids)
            elif self.metric == 'dot':
                new_centroids, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            return new_centroids, center_shift.item(), cluster_ids

        @torch.no_grad()
        def _compute_inertia(self, X: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor) -> float:
            """
            Computes the sum of squared distances of samples to their closest cluster center.
            Optimized version using vectorized operations.

            Parameters
            ----------
            X : torch.Tensor
                Input data of shape (n_samples, n_features).
            labels : torch.Tensor
                Cluster assignments of shape (n_samples,).
            centroids : torch.Tensor
                Cluster centers of shape (n_clusters, n_features).

            Returns
            -------
            float
                Total inertia.
            """
            # Get assigned centroids: (n_samples, n_features)
            assigned_centroids = centroids[labels]

            # Compute pairwise=False distances (element-wise comparison)
            dists = self.distance_metric(X, assigned_centroids, pairwise=False)

            return dists.sum().item()

        @torch.no_grad()
        def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
            """
            Performs k-means clustering on the input data and returns cluster labels.
            Optimized version without inertia computation during training.

            Parameters
            ----------
            X : torch.Tensor
                The input data of shape (n_samples, n_features).

            Returns
            -------
            torch.Tensor
                Cluster assignments of shape (n_samples,).
            """
            if X.ndim != 2:
                raise ValueError("X must be of shape (n_samples, n_features)")

            N, D = X.shape
            self.d = D  # Set feature dimensionality

            # Prepare data
            compute_dtype = self.dtype or X.dtype
            X = X.to(device=self.device, dtype=compute_dtype, copy=False)
            if self.metric == 'cosine':
                X = F.normalize(X, p=2, dim=-1)

            # Compute tolerance
            tol = torch.mean(torch.var(X, dim=0)).item() * self.tol

            min_shift = float('inf')  # Track minimum center shift instead of inertia
            best_centroids = None
            best_labels = None

            # Reset stats (optional: can be removed if not needed)
            self.stats = {'state': [], 'shift': [], 'label': []}

            # Multiple random initializations
            for n_init_idx in range(self.n_init):
                random_state = self.random_state + n_init_idx

                # Initialize centroids
                centroids = self.initialize(X, random_state=random_state)

                old_labels = None
                final_shift = float('inf')

                # Progress bar for this run
                progress_bar = tqdm.tqdm(total=self.max_iter, disable=not self.verbose)

                for n_iter in range(self.max_iter):
                    # Single iteration
                    new_centroids, center_shift, labels = self._single_iteration(X, centroids)

                    # Update progress
                    if self.verbose:
                        progress_bar.set_description(
                            f'n_init {n_init_idx + 1}/{self.n_init}, '
                            f'iter {n_iter}, shift {center_shift:.6f}'
                        )
                        progress_bar.update(1)

                    # Check for convergence
                    if old_labels is not None and torch.equal(labels, old_labels):
                        if self.verbose:
                            print(f"\nConverged at iteration {n_iter}: strict convergence.")
                        final_shift = center_shift
                        break
                    elif center_shift <= tol:
                        if self.verbose:
                            print(f"\nConverged at iteration {n_iter}: "
                                  f"center shift {center_shift:.2e} within tolerance {tol:.2e}.")
                        final_shift = center_shift
                        break

                    old_labels = labels.clone()
                    centroids = new_centroids
                    final_shift = center_shift

                progress_bar.close()

                # Store stats (using final shift instead of inertia)
                self.stats['state'].append(centroids)
                self.stats['shift'].append(final_shift)
                self.stats['label'].append(labels)

                # Track best result based on final center shift
                if final_shift < min_shift:
                    min_shift = final_shift
                    best_centroids = centroids
                    best_labels = labels

            # Convert stats to tensors
            self.stats['state'] = torch.stack(self.stats['state'])
            self.stats['shift'] = torch.tensor(self.stats['shift'])
            self.stats['label'] = torch.stack(self.stats['label'])

            if self.verbose:
                print(f"Final min center shift: {min_shift:.6f}")

            # Store final results
            self.cluster_centers_ = best_centroids
            self.labels_ = best_labels.long()
            self.inertia_ = None  # Set to None since we don't compute it during training

            # Set backward compatibility aliases
            self.centroids = self.cluster_centers_
            self.cluster_ids = self.labels_

            return self.labels_

        @torch.no_grad()
        def fit(self, X: torch.Tensor):
            """
            Fit k-means clustering on the input data.

            Alias for fit_predict that returns self for sklearn-style chaining.

            Parameters
            ----------
            X : torch.Tensor
                Input data of shape (n_samples, n_features).

            Returns
            -------
            self
                Fitted estimator.
            """
            self.fit_predict(X)
            return self

        @torch.no_grad()
        def train(self, X: torch.Tensor):
            """
            Fit k-means clustering on the input data.

            Backward compatibility method - same as fit().

            Parameters
            ----------
            X : torch.Tensor
                Input data of shape (n_samples, n_features).
            """
            self.fit_predict(X)

        @torch.no_grad()
        def predict(self, X: torch.Tensor, soft: bool = False) -> torch.Tensor:
            """
            Assigns samples to clusters based on fixed cluster centers.

            Parameters
            ----------
            X : torch.Tensor
                Input tensor of shape (n_samples, n_features).
            soft : bool, default=False
                If True, returns the soft assignment matrix (probabilities);
                if False, returns hard cluster assignments (indices).

            Returns
            -------
            torch.Tensor
                - If soft=False: (n_samples,) tensor of cluster indices.
                - If soft=True: (n_samples, n_clusters) tensor of probabilities.
            """
            if self.cluster_centers_ is None:
                raise RuntimeError("Model not trained. Call fit() or fit_predict() first.")

            if X.ndim != 2:
                raise ValueError("X must be of shape (n_samples, n_features)")

            N, D = X.shape
            if D != self.d:
                raise ValueError(f"Feature dimension mismatch: expected {self.d}, got {D}")

            # Prepare data
            compute_dtype = self.dtype or X.dtype
            X = X.to(device=self.device, dtype=compute_dtype, copy=False)

            dists = self.distance_metric(X, self.cluster_centers_)  # (n_samples, n_clusters)

            if soft:
                # Convert distances to probabilities
                # Smaller distance => higher probability
                return (-dists.sqrt()).softmax(dim=-1)
            else:
                # Hard assignment: return nearest cluster index
                return dists.argmin(dim=-1)

        @torch.no_grad()
        def transform(self, X: torch.Tensor) -> torch.Tensor:
            """
            Transform data to cluster-distance space.

            Parameters
            ----------
            X : torch.Tensor
                Shape: (n_samples, n_features)

            Returns
            -------
            torch.Tensor
                Distance to each cluster center. Shape: (n_samples, n_clusters)
            """
            if self.cluster_centers_ is None:
                raise RuntimeError("Model not trained. Call fit() or fit_predict() first.")

            if X.ndim != 2:
                raise ValueError("X must be of shape (n_samples, n_features)")

            N, D = X.shape
            if D != self.d:
                raise ValueError(f"Feature dimension mismatch: expected {self.d}, got {D}")

            compute_dtype = self.dtype or X.dtype
            X = X.to(device=self.device, dtype=compute_dtype, copy=False)

            # Compute distances in chunks
            split_size = min(4096, X.size(0))
            all_dists = []

            for chunk in X.split(split_size, dim=0):
                dists = self.distance_metric(chunk, self.cluster_centers_)
                all_dists.append(dists)

            return torch.cat(all_dists, dim=0)

        @torch.no_grad()
        def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
            """
            Fit k-means clustering and transform X to cluster-distance space.

            Parameters
            ----------
            X : torch.Tensor
                Input data of shape (n_samples, n_features).

            Returns
            -------
            torch.Tensor
                Distance to each cluster center. Shape: (n_samples, n_clusters)
            """
            self.fit_predict(X)
            return self.transform(X)

        @torch.no_grad()
        def score(self, X: torch.Tensor) -> float:
            """
            Compute the opposite of the value of X on the K-means objective.

            This method computes inertia on-demand, not during training.

            Parameters
            ----------
            X : torch.Tensor
                Input data of shape (n_samples, n_features).

            Returns
            -------
            float
                Opposite of the sum of squared distances of samples to their
                closest cluster center (negative inertia).
            """
            labels = self.predict(X, soft=False)
            inertia = self._compute_inertia(X, labels, self.cluster_centers_)
            return -inertia

        @torch.no_grad()
        def compute_inertia(self, X: torch.Tensor = None) -> float:
            """
            Compute inertia on-demand after training.

            Parameters
            ----------
            X : torch.Tensor, optional
                Input data. If None, uses the training labels stored in self.labels_.

            Returns
            -------
            float
                The inertia value.
            """
            if self.cluster_centers_ is None:
                raise RuntimeError("Model not trained. Call fit() or fit_predict() first.")

            if X is not None:
                labels = self.predict(X, soft=False)
            else:
                if self.labels_ is None:
                    raise RuntimeError("No training labels available. Provide X explicitly.")
                # Need to recompute using stored training data
                # Since we don't store X, user must provide it
                raise ValueError("X must be provided to compute inertia after training.")

            return self._compute_inertia(X, labels, self.cluster_centers_)

        def __repr__(self) -> str:
            """String representation of the TritonKMeans object."""
            return (
                f"TritonKMeans(metric={self.metric!r}, "
                f"init={self.init!r}, "
                f"n_clusters={self.n_clusters}, "
                f"n_init={self.n_init}, "
                f"max_iter={self.max_iter}, "
                f"tol={self.tol}, "
                f"random_state={self.random_state}, "
                f"verbose={self.verbose})"
            )
# ============================================================
# Fallback stub class when Triton is not available.
# Ensures Sphinx documentation builds and imports succeed
# without requiring Triton to be installed.
# ============================================================
else:
    class TritonKMeans:
        """
        Fast K-Means clustering implemented with Triton GPU kernels.

        This implementation provides an interface compatible with TorchKMeans
        while leveraging Triton kernels for improved performance.

        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric to use. Options: 'euclidean', 'cosine', 'dot'
        init : str or torch.Tensor, default='k-means++'
            Method for initialization: 'k-means++', 'random' or user-specified
            tensor of shape (n_clusters, n_features).
        random_state : int, optional
            Random seed for centroid initialization.
        n_clusters : int, default=8
            Number of clusters (k).
        n_init : int, default=10
            Number of times the algorithm will be run with different centroid seeds.
            The final result will be the best output of n_init consecutive runs.
        max_iter : int, default=300
            Maximum number of iterations for a single run.
        tol : float, default=1e-4
            Relative tolerance with regards to inertia to declare convergence.
        verbose : bool, default=False
            Whether to print per-iteration info.
        dtype : torch.dtype, optional
            Compute data type for algorithm.
        device : torch.device, optional
            Target device. Defaults to "cuda:0" when available.
            Currently, only CUDA devices are supported.
        distributed : bool, default=False
            Reserved for future distributed training support (currently not implemented).
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TritonKMeans requires the 'triton' package and a CUDA-capable GPU.\n"
                "Install it with:\n"
                "  pip install triton\n"
                "  # or\n"
                "  pip install pyagc[triton]\n\n"
                "Alternatively, use TorchKMeans for CPU/CUDA compatibility."
            )

        def fit(self, X):
            """Fit k-means clustering."""
            raise ImportError("triton is not installed.")

        def fit_predict(self, X):
            """Fit and predict cluster labels."""
            raise ImportError("triton is not installed.")

        def predict(self, X, soft=False):
            """Predict cluster labels."""
            raise ImportError("triton is not installed.")

        def transform(self, X):
            """Transform to cluster-distance space."""
            raise ImportError("triton is not installed.")

        def fit_transform(self, X):
            """Fit and transform."""
            raise ImportError("triton is not installed.")

        def __repr__(self) -> str:
            return "TritonKMeans(triton not installed)"
