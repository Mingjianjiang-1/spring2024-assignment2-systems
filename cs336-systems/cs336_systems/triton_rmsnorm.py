import torch 
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr : tl.pointer_type,
    weight_ptr : tl.pointer_type,
    x_row_stride : tl.uint32,
    output_ptr : tl.pointer_type,
    H : tl.uint32,
    BLOCK_SIZE: tl.constexpr):
    # Each instance will compute the weighted sum of a row of x.
    row_idx = tl.program_id(0)
    # Pointer to the first entry of the row this instance sums up.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    # Pointers to the entries we'll sum up.
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    # Load the data from x given the pointers to its entries,
    # using a mask since BLOCK_SIZE may be > H.
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = tl.sum(row * weight)
    # Write back output (a single scalar per instance).
    output_ptr = output_ptr + row_idx
    tl.store(output_ptr, output)



class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)

        H, output_dims = x.shape[-1], x.shape[:-1]

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(n_rows, )](
        x, weight, x.stride(0), y, H,
        num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y


############################################################################################################
# Write a Triton kernel that computes the forward pass of the RMSNorm operation.
############################################################################################################

class PytorchRMSNorm(torch.autograd.Function):
    def jvp_g(grad_output, x, weight, epsilon):
        
        new_x = grad_output * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * x # Shape (*, *, H)
        
        return new_x.sum(dim=tuple(range(new_x.ndim - 1)), keepdim=True)
    
    def jvp_x(grad_output, x, weight, epsilon):
        
        H = x.shape[-1]
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        z = (1 / H) * x * weight * (rms**3)
        w = weight * rms
        g_x1 = grad_output * w
        g_x2 = x * (z * grad_output).sum(dim=-1, keepdim=True)
        
        return g_x1 - g_x2   
    
    @staticmethod
    def forward(ctx, x, weight, epsilon=1e-5):
        
        # x (Tensor): Input tensor of shape (*, *, H)
        # weight (Tensor): Weight tensor of shape (H,)
        
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) # Shape (*, *, 1)
        x = x * rms # Shape (*, *, H)
        
        return weight * x  
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        epsilon = ctx.epsilon
        
        grad_x = PytorchRMSNorm.jvp_x(grad_output, x, weight, epsilon)
        grad_weight = PytorchRMSNorm.jvp_g(grad_output, x, weight, epsilon)
        
        return grad_x, grad_weight

        
@triton.jit
def rmsnorm_fwd(
    x_ptr : tl.pointer_type,
    weight_ptr : tl.pointer_type,
    output_ptr : tl.pointer_type,
    H : tl.uint32,
    epsilon : tl.float32,
    BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0)
 
    row_start_ptr = x_ptr + row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    mask = offsets < H	
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = row / tl.sqrt(tl.sum(row * row) / H + epsilon) * weight
    
    output_ptr = output_ptr + row_idx * H
    tl.store(output_ptr, output, mask=mask)


@triton.jit
def rmsnorm_bwd(x_ptr : tl.pointer_type,
                 g_ptr : tl.pointer_type, 
                 dout_ptr : tl.pointer_type,
                 dx_ptr : tl.pointer_type,
                 dg_ptr : tl.pointer_type,
                 H : tl.uint32,
                 eps: tl.float32,
                 BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    
    x_start_ptr = x_ptr + row_idx * H
    dout_start_ptr = dout_ptr + row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    xi_val = tl.load(x_start_ptr + offsets, mask=mask, other=0)
    g_val = tl.load(g_ptr + offsets, mask=mask, other=0)
    dout = tl.load(dout_start_ptr + offsets, mask=mask, other=0)
    
    denum = tl.sqrt(tl.sum(xi_val * xi_val) / H + eps)
    dg = dout * (xi_val/denum)
    dx = dout * (g_val/denum) - xi_val * tl.sum(dout * g_val * xi_val / (denum*denum*denum)) / H

    # store the results
    dx_start_ptr = dx_ptr + row_idx * H
    dg_start_ptr = dg_ptr + row_idx * H
    tl.store(dx_start_ptr + offsets, dx, mask=mask)
    tl.store(dg_start_ptr + offsets, dg, mask=mask)
    

class TritonRMSNormKernel(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, epsilon=1e-5):
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        H, output_dims = x.shape[-1], x.shape
        x = x.view(-1, H)
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(output_dims, device=x.device)
  
        # Launch our kernel with n_rows instances in our 1D grid.
        n_rows = x.shape[0]  # Correcting how n_rows is determined
        rmsnorm_fwd[(n_rows, )](
            x, weight, x.stride(0), y, H, epsilon,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=16)

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        epsilon = ctx.epsilon
        H = x.shape[-1]
        
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        
        n_rows = int(grad_output.numel() / H)
        rmsnorm_bwd[(n_rows, )](
            x, weight, grad_output, grad_x, grad_weight, H, epsilon,
            BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=16)
        
        grad_weight = grad_weight.sum(dim=tuple(range(grad_weight.ndim - 1)), keepdim=True)
        
        return grad_x, grad_weight


 