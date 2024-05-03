import torch
from cs336_basics.model import BasicsTransformerLM, RMSNorm
from cs336_basics.optimizer import AdamW
import numpy as np
import os
import logging
import timeit
import argparse
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import gc

# Setup logging
def benchmark(model, data, optimizer, config, model_type='lm'):
    scaler = GradScaler() if config['mixed_precision'] else None

    # Warm up
    for _ in range(config['num_warmups']):
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1))   
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config['num_trials']):
        start_time = timeit.default_timer()
        context = autocast() if config['mixed_precision'] else nullcontext()
        with context:
            logits = model(data)
            if model_type == 'lm':
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1))
        
        if config['include_backward']:
            if config['mixed_precision']:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()

        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time


def benchmark_lm(model_size, config):
    filename = 'benchmarking_results/lm/benchmarking_backward{include_backward}_mixed{mixed_precision}.log'.format(include_backward=config['include_backward'], mixed_precision=config['mixed_precision'])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(filename)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    context_length = 128
    vocab_size = 10000
    
    logging.info(f"Running benchmark for {model_size} model with {'mixed precision' if config['mixed_precision'] else 'full precision'}")
    logging.info(f"Configuration: {config}")

    # Load the model with specific size parameters
    # Empty the cache to avoid OOM errors
    torch.cuda.empty_cache()
    gc.collect()
    
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        attn_pdrop=0.1,
        residual_pdrop=0.1,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])
    data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    
    mean_time, std_time = benchmark(model, data, optimizer, config)
    # Log the results
    logging.info(f"Benchmarking {model_size} model with {'mixed precision' if config['mixed_precision'] else 'full precision'}")
    logging.info("Mean time: %.2f ms (std: %.2f ms)", mean_time, std_time)
    logging.info("")
    

def benchmark_layernorm(config):
    filename = 'benchmarking_results/ln/benchmarking_dmodel{d_ln}.log'.format(d_ln=config['d_ln'])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(filename)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(50000, config['d_ln'], device=device)
    
    model = torch.nn.LayerNorm(config['d_ln']).to(device)
    # Random initialize the model
    torch.nn.init.normal_(model.weight)
    torch.nn.init.normal_(model.bias)
    
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    mean_time, std_time = benchmark(model, data, optimizer, config, model_type='ln')
    # Log the results
    logging.info(f"Benchmarking LayerNorm")
    logging.info("Mean time: %.2f ms (std: %.2f ms)", mean_time, std_time)
    
    model = RMSNorm(config['d_ln']).to(device)
    # Random initialize the model
    torch.nn.init.normal_(model.weight)
    
    mean_time, std_time = benchmark(model, data, optimizer, config, model_type='ln')
    # Log the results
    logging.info(f"Benchmarking RMSNorm")
    logging.info("Mean time: %.2f ms (std: %.2f ms)", mean_time, std_time)
    logging.info("")
    

def run_experiments_lm(args):
    # Default configuration
    default_config = {
        'lr': 1e-3,
        'num_warmups': 1,
        'num_trials': 5,
        'include_backward': args.include_backward,
        'mixed_precision': args.mixed_precision,
        # Add any other default values here
    }

    # Specific configurations for each model size
    model_configs = {
        'small': {'num_layers': 12, 'num_heads': 12, 'd_model': 768, 'd_ff': 3072},
        'medium': {'num_layers': 24, 'num_heads': 16, 'd_model': 1024, 'd_ff': 4096},
        'large': {'num_layers': 36, 'num_heads': 20, 'd_model': 1280, 'd_ff': 5120},
        'xl': {'num_layers': 48, 'num_heads': 25, 'd_model': 1600, 'd_ff': 6400},
        # '2.7B': {'num_layers': 32, 'num_heads': 32, 'd_model': 2560, 'd_ff': 10240},
        # Add other model sizes with their specific configurations
    }

    for model_size, specific_config in model_configs.items():
        # Copy the default config and update it with the specific configurations
        config = default_config.copy()
        config.update(specific_config)
        benchmark_lm(model_size, config)

def run_experiments_ln():
    # Default configuration
    default_config = {
        'lr': 1e-3,
        'num_warmups': 1,
        'num_trials': 1000,
        'include_backward': False,
        'mixed_precision': False,
        # Add any other default values here
    }

    d_lns = [1024, 2048, 4096, 8192]
    
    for d_ln in d_lns:
        # Copy the default config and update it with the specific configurations
        config = default_config.copy()
        config['d_ln'] = d_ln
        benchmark_layernorm(d_ln, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark the model with different configurations')
    parser.add_argument('--include_backward', action='store_true', help='Include the backward pass in the benchmark')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision in the benchmark')
    parser.add_argument('--layernorm', action='store_true')
    args = parser.parse_args()
    if args.layernorm:
        run_experiments_ln()
    else:
        run_experiments_lm(args)
    
