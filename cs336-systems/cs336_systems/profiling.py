import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import numpy as np
import os
import logging
import timeit
from torch.profiler import profile, record_function, ProfilerActivity


# Setup logging
# logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

def run_step(model, inputs, optimizer, enable_backward, vocab_size=10000):
	with record_function('forward_pass'):
		model.forward(inputs)

	if enable_backward:
		with record_function('backward_pass'):
			logits = model(inputs)
			loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), inputs.view(-1))
			loss.backward()
		with record_function('optimizer'):
			optimizer.step()
			optimizer.zero_grad(set_to_none=True)

def _profile(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    context_length = 128
    vocab_size = 10000
    
    logging.info(f"Running profile")

    # Load the model with specific size parameters
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=32,
        num_heads=32,
        d_model=2560,
        d_ff=10240,
        attn_pdrop=0.1,
        residual_pdrop=0.1,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])
    data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # Warm up
    for _ in range(config['num_warmups']):
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()

    torch.cuda.synchronize()
    
    with profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for _ in range(config['num_trials']):
            run_step(model, data, optimizer, config['include_backward'], vocab_size=vocab_size)
            prof.step()

    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    logging.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    
def run_experiments(include_backward):
    # Default configuration
    default_config = {
        'lr': 1e-3,
        'num_warmups': 1,
        'num_trials': 5,
        'include_backward': include_backward,
        'num_layers': 48, 'num_heads': 25, 'd_model': 1600, 'd_ff': 6400,
        # Add any other default values here
    }

    # Run the default configuration
    _profile(default_config)
    

if __name__ == "__main__":
    include_backward = False
    suffix = ''
    if include_backward:
        suffix = '_with_backward'
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(f'profiling{suffix}.log'), logging.StreamHandler()])
    run_experiments(include_backward=include_backward)
    # run_experiments(include_backward=True)
