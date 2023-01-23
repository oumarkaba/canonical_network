import ray
import torch
# from local_ot_emd import emd as ot_emd
import ot

# ot.emd
# import emd_testing
import numpy as np
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.lp.emd_wrap import emd_c


@ray.remote
def ot_emd(a, b, M, i, numItermax=100000, numThreads=1):
    # convert to numpy if list
    a, b, M = a[i], b[i], M[i]
    a, b, M = list_to_array(a, b, M)

    a0, b0, M0 = a, b, M
    nx = get_backend(M0, a0, b0)

    # convert to numpy
    M = nx.to_numpy(M)
    a = nx.to_numpy(a)
    b = nx.to_numpy(b)

    # ensure float64
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64, order='C')

    b = b * a.sum() / b.sum()

    G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)

    return nx.from_numpy(G, type_as=M0), nx.from_numpy(u, type_as=a0), nx.from_numpy(v, type_as=b0)

# @ray.remote
# def single_emd(a,b,M,i):
#     return ot_emd(a[i], b[i], M[i])

def batch_ray_emd(a,b,M):
    a_ = a.detach().cpu()
    b_ = b.detach().cpu()
    M_ = M.detach().cpu()
    a_id = ray.put(a_)
    b_id = ray.put(b_)
    M_id = ray.put(M_)
    results = [ot_emd.remote(a_id, b_id, M_id, i) for i in range(a.size(0))]
    # return [ray.get(r) for r in results]
    results = zip(*[ray.get(r) for r in results])
    gamma, u, v = [torch.stack(res).to(device=M.device, dtype=M.dtype) for res in results]
    return gamma, u, v

def batch_diag(input):
    bsize, dim, device = input.shape[0], input.shape[-1], input.device
    out = torch.zeros(*input.shape[:-1], dim, dim, 
        device=device, dtype=input.dtype)
    
    diag_idx = torch.arange(dim, device=device).expand(bsize, -1).flatten()
    batch_idx = torch.arange(bsize, device=device).expand(dim, -1).T.flatten()
    out[batch_idx, diag_idx, diag_idx] = input.flatten()
    return out

def get_optimal_transport(num_sink, lambda_sh, ret_type, uv_grad=False,
    lambda_a_grad=1, lambda_b_grad=1, lambda_p_dis=1, lambda_p_sh=1,
    center_ab_grad=False, center_p_grad=False, fixed_p=None, cg_iters=10):
    assert ret_type in ['emd', 'emd_plus', 'sh', 'fixed'], ret_type
    assert not uv_grad or 'emd' in ret_type

    def sinkhorn_fwd(c,a,b):
        log_p = -c / lambda_sh
        log_a = torch.log(a).unsqueeze(dim=2)
        log_b = torch.log(b).unsqueeze(dim=1)
        for _ in range(num_sink):
            log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=2, keepdim=True) - log_a)
        p = torch.exp(log_p)
        return p
    
    def sinkhorn_bwd_ab(p, a, b, grad_p):
        bsize, m, n = p.shape
        K = torch.cat((
            torch.cat((batch_diag(a), p), dim=2),
            torch.cat((p.transpose(1,2), batch_diag(b)), dim=2)),
            dim=1)[:,:-1, :-1]
        t = torch.cat((
            grad_p.sum(dim=2),
            grad_p[:, :, :-1].sum(dim=1)),
            dim=1).unsqueeze(2)
        grad_ab = conjugate_gradient(lambda v: K@v, t, t, cg_iters)
        # grad_ab  = torch.linalg.solve(K, t)
        grad_a = grad_ab[:,:m, :]
        grad_b = torch.cat((grad_ab[:,m:, :], torch.zeros([bsize, 1, 1],
            device=p.device, dtype=torch.float32)), dim=1)
        return grad_a, grad_b

    class OptimalTransport(torch.autograd.Function):
        @staticmethod
        def forward(ctx, c, a, b):
            p_sh = sinkhorn_fwd(c, a, b)
            if ret_type == 'emd':
                p, u, v = batch_ray_emd(a,b,c)
            elif ret_type == 'emd_plus':
                p, u, v = batch_ray_emd(a,b,c)
                p = lambda_p_dis * p + lambda_p_sh * p_sh
            elif ret_type == 'fixed':
                p, u, v = fixed_p, None, None
            else:
                p = p_sh
                u = v = None
            ctx.save_for_backward(p_sh, torch.sum(p_sh, dim=2), torch.sum(p_sh, dim=1), u, v)
            return p, p_sh

        @staticmethod
        def backward(ctx, grad_p, grad_p_sh):
            p, a, b, u, v = ctx.saved_tensors
            grad_p = grad_p * -1 / lambda_sh * p
            grad_a, grad_b = sinkhorn_bwd_ab(p, a, b, grad_p)
            U = grad_a + grad_b.transpose(1,2)
            grad_p -= p * U
            if uv_grad:
                grad_a = u.unsqueeze(2)
                grad_b = v.unsqueeze(2)
            if center_p_grad:
                grad_p -= grad_p.mean(dim=(1,2), keepdim=True)
            if center_ab_grad:
                grad_a -= grad_a.mean(1, keepdim=True)
                grad_b -= grad_b.mean(1, keepdim=True)
            grad_a = -lambda_sh * grad_a.squeeze(dim=2) * lambda_a_grad
            grad_b = -lambda_sh * grad_b.squeeze(dim=2) * lambda_b_grad
            return grad_p, grad_a, grad_b
    return OptimalTransport.apply

def conjugate_gradient(hvp, x_init, b, iters=10):
    x = x_init
    r = b - hvp(x)
    p = r
    bdot = lambda a, b: torch.einsum('ncd, ncd -> n', a, b).clamp(min=1e-12)
    for i in range(iters):
        Ap = hvp(p)
        alpha = bdot(r, r) / bdot(p, Ap)
        alpha = alpha[:,None,None]
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = bdot(r_new, r_new) / bdot(r, r)
        beta = beta[:,None,None]
        p = r_new + beta * p
        r = r_new
    return x