from dataclasses import dataclass
from typing import List, Dict, Any
import itertools

@dataclass
class ModelConfig:
    embed_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    max_seq_len: int
    dropout: float
    batch_size: int
    learning_rate: float

def generate_grid_search_configs() -> List[ModelConfig]:
    """
    Generate all possible combinations of model configurations for grid search
    """
    param_grid = {
        'embed_dim': [64, 128, 256],
        'num_layers': [4, 8],
        'num_heads': [4, 8],
        'ff_dim': [256, 512, 1024],
        'dropout': [0.1, 0.2],
        'learning_rate': [1e-4, 5e-4]
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Convert to ModelConfig objects
    configs = []
    for combo in combinations:
        config = ModelConfig(
            embed_dim=combo[0],
            num_layers=combo[1],
            num_heads=combo[2],
            ff_dim=combo[3],
            dropout=combo[4],
            learning_rate=combo[5]
        )
        configs.append(config)
    
    return configs

def get_config_name(config: ModelConfig) -> str:
    """
    Generate a unique name for the configuration
    """
    return f"model_emb{config.embed_dim}_layers{config.num_layers}_heads{config.num_heads}_ff{config.ff_dim}_drop{config.dropout}_lr{config.learning_rate}" 