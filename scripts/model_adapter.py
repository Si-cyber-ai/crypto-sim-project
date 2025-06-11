import torch
import torch.nn as nn
from model_architecture import TransformerModel

def adapt_old_state_dict(old_state_dict, new_model):
    """
    Adapt old state dict to new model architecture.
    
    Args:
        old_state_dict: The state dict from the old model
        new_model: An instance of the new model architecture
        
    Returns:
        new_state_dict: A state dict compatible with the new model
    """
    new_state_dict = new_model.state_dict()
    
    # Map old keys to new keys
    key_mapping = {
        'input_linear.weight': 'embedding.weight',
        'input_linear.bias': 'embedding.bias',
        'output_linear.weight': 'fc_out.weight',
        'output_linear.bias': 'fc_out.bias'
    }
    
    # Copy transformer encoder layers directly (they have the same names)
    for key in old_state_dict:
        if key.startswith('transformer_encoder'):
            if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
                new_state_dict[key] = old_state_dict[key]
    
    # Copy and rename the input/output layers
    for old_key, new_key in key_mapping.items():
        if old_key in old_state_dict and new_key in new_state_dict:
            if old_state_dict[old_key].shape == new_state_dict[new_key].shape:
                new_state_dict[new_key] = old_state_dict[old_key]
            else:
                print(f"Shape mismatch for {old_key} -> {new_key}: {old_state_dict[old_key].shape} vs {new_state_dict[new_key].shape}")
    
    return new_state_dict

def optimize_model_for_inference(model):
    """
    Optimize the model for inference by applying torch.compile if available.
    
    Args:
        model: The PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    # Check if torch.compile is available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            # Use torch.compile for faster inference
            model = torch.compile(model)
            print("Model optimized with torch.compile")
        except Exception as e:
            print(f"Could not optimize model with torch.compile: {e}")
    
    return model

def create_fx_graph(model):
    """
    Create an FX graph from the model for analysis and optimization.
    
    Args:
        model: The PyTorch model
        
    Returns:
        GraphModule or None if tracing fails
    """
    try:
        # Try to create an FX graph
        from torch.fx import symbolic_trace
        
        # Trace the model
        traced_model = symbolic_trace(model)
        
        # Print the graph for debugging
        print("FX Graph:")
        print(traced_model.graph)
        
        return traced_model
    except Exception as e:
        print(f"Could not create FX graph: {e}")
        return None

def analyze_model_performance(model, sample_input):
    """
    Analyze model performance to identify bottlenecks.
    
    Args:
        model: The PyTorch model
        sample_input: Sample input tensor
        
    Returns:
        None, prints analysis results
    """
    try:
        # Check if torch.profiler is available
        if hasattr(torch, 'profiler'):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Run the model with the sample input
                model(sample_input)
            
            # Print profiling results
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            # Fallback to simple timing
            import time
            start_time = time.time()
            model(sample_input)
            end_time = time.time()
            print(f"Model inference time: {(end_time - start_time) * 1000:.2f} ms")
    except Exception as e:
        print(f"Could not analyze model performance: {e}")
