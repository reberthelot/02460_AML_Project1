import torch

def append_parameters_to_checkpoint(params,checkpoint_path: str):
    """
    Hardcoded function to open an existing .pt checkpoint, 
    append or modify parameters, and save it back.
    """
    # Hardcoded parameters to add/update
    

    try:
        # Load the existing checkpoint
        # Using map_location='cpu' to ensure it opens regardless of original device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if not isinstance(checkpoint, dict):
            print("Error: Checkpoint is not a dictionary. Creating a new structure.")
            checkpoint = {'model_state_dict': checkpoint}
        else :
            for k,v in checkpoint.items():
                if k != 'model_state_dict':
                    print(f"{k} = {v}")
        # Append/Update the parameters
        for key, value in params.items():
            if not(checkpoint.get(key)):
                checkpoint[key] = value
                print(f"Appended: {key} = {value}")

        # Save the modified checkpoint back to the file
        torch.save(checkpoint, checkpoint_path)
        print(f"Successfully updated checkpoint at: {checkpoint_path}")

    except FileNotFoundError:
        print(f"Error: The file {checkpoint_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extra_params = {
        'beta_vae': 'results_beta_flow/model_flow_beta_1e-06.pt',
    }
    path = 'output_PartB/model_ddpm_bvae_unet.pt'
    append_parameters_to_checkpoint(extra_params,path)
