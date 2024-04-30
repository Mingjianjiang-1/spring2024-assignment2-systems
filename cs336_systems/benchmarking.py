import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import numpy as np
import os
import logging
import timeit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s', filemode='w', filename='benchmarking.log')

def benchmark_model(model_size, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    context_length = 128
    vocab_size = 10000
    
    logging.info(f"Running benchmark for {model_size} model")
    logging.info(f"Configuration: {config}")

    # Load the model with specific size parameters
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

    # Warm up
    for _ in range(config['num_warmups']):
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), data.view(-1))
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config['num_trials']):
        start_time = timeit.default_timer()
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), data.view(-1))

        if config['include_backward']:
            loss.backward()

        torch.cuda.synchronize()

        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)

    # Log the results
    logging.info(f"Benchmarking {model_size} model")
    logging.info("Mean time: %.2f ms (std: %.2f ms)", mean_time, std_time)
    logging.info("")
    
    
def run_experiments(include_backward):
    # Default configuration
    default_config = {
        'lr': 1e-3,
        'num_warmups': 1,
        'num_trials': 5,
        'include_backward': include_backward,
        # Add any other default values here
    }

    # Specific configurations for each model size
    model_configs = {
        'small': {'num_layers': 12, 'num_heads': 12, 'd_model': 768, 'd_ff': 3072},
        'medium': {'num_layers': 24, 'num_heads': 16, 'd_model': 1024, 'd_ff': 4096},
        'large': {'num_layers': 36, 'num_heads': 20, 'd_model': 1280, 'd_ff': 5120},
        'xl': {'num_layers': 48, 'num_heads': 25, 'd_model': 1600, 'd_ff': 6400},
        '2.7B': {'num_layers': 32, 'num_heads': 32, 'd_model': 2560, 'd_ff': 10240},
        # Add other model sizes with their specific configurations
    }

    for model_size, specific_config in model_configs.items():
        # Copy the default config and update it with the specific configurations
        config = default_config.copy()
        config.update(specific_config)
        benchmark_model(model_size, config)

if __name__ == "__main__":
    
    run_experiments(include_backward=False)
    run_experiments(include_backward=True)
