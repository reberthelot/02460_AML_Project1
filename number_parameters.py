import torch
import argparse
import os

def count_parameters(model):
    return sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))

def main():
    parser = argparse.ArgumentParser(description="Count parameters of a saved model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model file")
    parser.add_argument("--device", type=str, default='cpu', help="Device to load model on")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: File {args.model} not found.")
        return

    try:
        checkpoint = torch.load(args.model, map_location=args.device)
        
        # Handle both full save dicts and raw state dicts
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        params = count_parameters(state_dict)

        print("-" * 30)
        print(f"Model Path: {args.model}")
        print(f"Total Parameters in State Dict: {params:,}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred while reading the checkpoint: {e}")
        print("Ensure the file is a valid PyTorch checkpoint.")

if __name__ == "__main__":
    main()
