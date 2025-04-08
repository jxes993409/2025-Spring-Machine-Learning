import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, Qwen2Config, set_seed
from datasets import load_dataset
from typing import Dict, Any, Optional

# Define Dataset
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset

class PixelSequenceDataset(Dataset):
	def __init__(self, data: List[List[int]], mode: str = "train"):
		"""
		A dataset class for handling pixel sequences.

		Args:
			data (List[List[int]]): A list of sequences, where each sequence is a list of integers.
			mode (str): The mode of operation, either "train", "dev", or "test".
				- "train": Returns (input_ids, labels) where input_ids are sequence[:-1] and labels are sequence[1:].
				- "dev": Returns (input_ids, labels) where input_ids are sequence[:-160] and labels are sequence[-160:].
				- "test": Returns only input_ids, as labels are not available.
		"""
		self.data = data
		self.mode = mode

	def __len__(self) -> int:
		"""Returns the total number of sequences in the dataset."""
		return len(self.data)

	def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
		"""
		Fetches a sequence from the dataset and processes it based on the mode.

		Args:
			idx (int): The index of the sequence.

		Returns:
			- If mode == "train": Tuple[torch.Tensor, torch.Tensor] -> (input_ids, labels)
			- If mode == "dev": Tuple[torch.Tensor, torch.Tensor] -> (input_ids, labels)
			- If mode == "test": torch.Tensor -> input_ids
		"""
		sequence = self.data[idx]

		if self.mode == "train":
			input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
			labels = torch.tensor(sequence[1:], dtype=torch.long)
			return input_ids, labels

		elif self.mode == "dev":
			input_ids = torch.tensor(sequence[:-160], dtype=torch.long)
			labels = torch.tensor(sequence[-160:], dtype=torch.long)
			return input_ids, labels

		elif self.mode == "test":
			input_ids = torch.tensor(sequence, dtype=torch.long)
			return input_ids

		raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'dev', or 'test'.")

# Visualization
def pixel_to_image(pixel_color: List[int], colormap: List[List[int]]) -> Image.Image:
    """
    Converts a list of pixel indices into a 20x20 RGB image using a colormap.

    Args:
        pixel_color (List[int]): A list of pixel indices representing colors.
        colormap (List[List[int]]): A list where each index maps to an RGB color [R, G, B].

    Returns:
        Image.Image: A PIL Image object representing the reconstructed image.
    """
    # Ensure the pixel_color list has at least 400 elements (pad with 0s if needed)
    while len(pixel_color) < 400:
        pixel_color.append(0)

    # Map pixel indices to actual RGB colors using the colormap
    pixel_data = [colormap[pixel] for pixel in pixel_color]

    # Convert to numpy array and reshape to 20x20x3 (RGB image)
    image_array = np.array(pixel_data, dtype=np.uint8).reshape(20, 20, 3)

    # Create a PIL Image from the array
    image = Image.fromarray(image_array)

    return image

def show_images(images: List[Image.Image]) -> None:
    """
    Displays a grid of up to 96 images using Matplotlib.

    Args:
        images (List[Image.Image]): A list of PIL Image objects to display.

    Returns:
        None
    """
    num_images = min(96, len(images))  # Limit to 96 images

    # Set up the figure size and grid layout (6 rows, 16 columns)
    fig, axes = plt.subplots(6, 16, figsize=(16, 6))
    axes = axes.flatten()  # Flatten to make iteration easier

    # Loop through images and display each one in the grid
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')  # Hide axis
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Save Model Function
def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, save_dir: str, filename: str = "best_model.pth") -> None:
    """
    Saves the model state, optimizer state, current epoch, and loss to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
        epoch (int): The current epoch number (used for checkpointing).
        loss (float): The current loss value to track model performance.
        save_dir (str): The directory where the model checkpoint will be saved.
        filename (str, optional): The name of the file to save the model. Defaults to "best_model.pth".

    Returns:
        None
    """
    # Construct the full path for saving the model checkpoint
    save_path = os.path.join(save_dir, filename)

    # Save the model, optimizer state, and additional metadata (epoch and loss)
    torch.save({
        'epoch': epoch + 1,                # Save epoch + 1 for easier tracking
        'model_state_dict': model.state_dict(),       # Save model weights
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state (important for resuming training)
        'loss': loss                     # Save the current loss value
    }, save_path)

    # Print a confirmation message indicating the model has been saved
    print(f"Model saved at {save_path} (Loss: {loss:.4f}, Epoch: {epoch + 1})")

# Output parameter
def print_parameter(batch_size: int, epochs: int, learning_rate: float, config: Dict[str, Any]) -> None:

	output_file: str = "parameters.txt"

	with open(output_file, "w") as f:
		# Write each sequence to the file
		print(f"batch size: {batch_size}", file=f)
		print(f"epoch: {epochs}", file=f)
		print(f"lr: {learning_rate:.1e}", file=f)
		print("config = {", file=f)
		for k, v in config.items():
			print(f"\t{k}: {v},", file=f)
		print("}", file=f)

if __name__ == "__main__":
	# Set Random Seed
	set_seed(0)
	# Load the pokemon dataset from Hugging Face Hub
	pokemon_dataset = load_dataset("lca0503/ml2025-hw4-pokemon")

	# Load the colormap from Hugging Face Hub
	colormap = list(load_dataset("lca0503/ml2025-hw4-colormap")["train"]["color"])

	# Define number of classes
	num_classes = len(colormap)

	# Define batch size
	batch_size = 32

	# === Prepare Dataset and DataLoader for Training ===
	train_dataset: PixelSequenceDataset = PixelSequenceDataset(
		pokemon_dataset["train"]["pixel_color"], mode="train"
	)
	train_dataloader: DataLoader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True
	)

	# === Prepare Dataset and DataLoader for Validation ===
	dev_dataset: PixelSequenceDataset = PixelSequenceDataset(
		pokemon_dataset["dev"]["pixel_color"], mode="dev"
	)
	dev_dataloader: DataLoader = DataLoader(
		dev_dataset, batch_size=batch_size, shuffle=False
	)

	# === Prepare Dataset and DataLoader for Testing ===
	test_dataset: PixelSequenceDataset = PixelSequenceDataset(
		pokemon_dataset["test"]["pixel_color"], mode="test"
	)
	test_dataloader: DataLoader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False
	)

	# Model Configuration

	# Define GPT-2 model configuration as a dictionary
	gpt2_config = {
		"activation_function": "gelu_new",    # Activation function used in the model
		"architectures": ["GPT2LMHeadModel"],  # Specifies the model type
		"attn_pdrop": 0.1,            # Dropout rate for attention layers
		"embd_pdrop": 0.1,            # Dropout rate for embeddings
		"initializer_range": 0.02,        # Standard deviation for weight initialization
		"layer_norm_epsilon": 1e-05,       # Small constant to improve numerical stability in layer norm
		"model_type": "gpt2",           # Type of model
		"n_ctx": 128,               # Context size (maximum sequence length)
		"n_embd": 64,              # Embedding size
		"n_head": 2,               # Number of attention heads
		"n_layer": 2,              # Number of transformer layers
		"n_positions": 400,           # Maximum number of token positions
		"resid_pdrop": 0.1,           # Dropout rate for residual connections
		"vocab_size": num_classes,       # Number of unique tokens in vocabulary
		"pad_token_id": None,          # Padding token ID (None means no padding token)
		"eos_token_id": None,          # End-of-sequence token ID (None means not explicitly defined)
	}

	# Define Qwen-2 model configuration as a dictionary
	qwen2_config = {
		"vocab_size": num_classes,
		"hidden_size": 512,
		"intermediate_size": 896,
		"num_hidden_layers": 4,
		"num_attention_heads": 32,
		"num_key_value_heads": 32,
		"hidden_act": "gelu_new",
		"max_position_embeddings": 400,
		"rms_norm_eps": 1e-05,
		"use_sliding_window": False,
		"attention_dropout": 0.1,
		"pad_token_id": None,
		"eos_token_id": None,
	}


	# Load GPT-2 model configuration from dictionary
	# config_gpt2 = GPT2Config.from_dict(gpt2_config)
	config_qwen2 = Qwen2Config.from_dict(qwen2_config)

	config = config_qwen2

	# Load the model using the configuration defined above
	model = AutoModelForCausalLM.from_config(config)

	print(model)

	# Count trainable parameters
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Trainable Parameters: {trainable_params:,}")

	# Training Arguments

	epochs = 100															# Number of training epochs
	learning_rate = 1e-3													# Learning rate for optimizer
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	# Check if CUDA is available for GPU
	save_dir = "checkpoints"												# Directory to save model checkpoints

	# Loss function and optimizer
	criterion = nn.CrossEntropyLoss()												# Loss function for classification tasks
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)	# AdamW optimizer with weight decay

	# print_parameter(batch_size, epochs, learning_rate, qwen2_config)

	# Create save directory if it doesn't exist
	os.makedirs(save_dir, exist_ok=True)
	# Initialize best loss as positive infinity for comparison during model checkpointing
	best_loss: float = float('inf')
	# Move model to the appropriate device (GPU or CPU)
	model.to(device)

	# Training Loop
	for epoch in range(epochs):
		model.train()	# Set the model to training mode
		epoch_loss = 0	# Initialize the epoch loss

		# Iterate over training data batches
		for input_ids, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
			input_ids, labels = input_ids.to(device), labels.to(device)  # Move data to the same device as the model

			# Forward pass through the model to get logits (output probabilities)
			outputs = model(input_ids=input_ids).logits.view(-1, config.vocab_size)
			labels = labels.view(-1)  # Flatten labels to match logits shape

			# Calculate loss using CrossEntropyLoss
			loss = criterion(outputs, labels)

			# Backpropagation and optimizer step
			optimizer.zero_grad()	# Reset gradients to zero
			loss.backward()			# Compute gradients
			optimizer.step()		# Update model weights

			# Accumulate the loss for the epoch
			epoch_loss += loss.item()

		# Compute average epoch loss
		avg_epoch_loss = epoch_loss / len(train_dataloader)
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

		# Evaluation Loop (Validation)
		model.eval()		# Set the model to evaluation mode (disables dropout, etc.)
		total_accuracy = 0	# Initialize total accuracy
		num_batches = 0		# Initialize batch counter

		with torch.no_grad():  # Disable gradient calculation for validation
			# Iterate over validation data batches
			for inputs, labels in tqdm(dev_dataloader, desc="Evaluating"):
				inputs, labels = inputs.to(device), labels.to(device)	# Move validation data to device
				attention_mask = torch.ones_like(inputs)				# Attention mask to ensure valid token positions

				# Perform batch inference using the model
				generated_outputs = model.generate(inputs, attention_mask=attention_mask, max_length=400)

				# Extract the last 160 tokens from generated outputs and labels
				generated_outputs = generated_outputs[:, -160:]

				# Calculate accuracy for the batch
				accuracy = (generated_outputs == labels).float().mean().item()
				total_accuracy += accuracy
				num_batches += 1

		# Compute average reconstruction accuracy for the epoch
		avg_accuracy = total_accuracy / num_batches
		print(f"Epoch {epoch + 1}/{epochs}, Reconstruction Accuracy: {avg_accuracy:.4f}")

		# If the current epoch loss is better (lower) than the best loss, save the model
		if avg_epoch_loss < best_loss:
			best_loss = avg_epoch_loss	# Update best loss
			save_model(model, optimizer, epoch, best_loss, save_dir)  # Save the model with the best loss

	# Load the best model from the saved checkpoint
	best_model_path = os.path.join(save_dir, "best_model.pth")							# Path to the best model checkpoint
	checkpoint = torch.load(best_model_path, weights_only=True, map_location=device)	# Load checkpoint from the file
	model.load_state_dict(checkpoint["model_state_dict"])								# Load the model weights from checkpoint
	model.eval()																		# Set the model to evaluation mode (disables dropout, etc.)

	# Inference

	# Testing Loop with Batch Inference
	results: list = []	# List to store the generated sequences from the model

	with torch.no_grad():	# Disable gradient calculations for inference
		# Iterate over test data in batches
		for inputs in tqdm(test_dataloader, desc="Generating Outputs"):
			inputs = inputs.to(device)					# Move model to the appropriate device (GPU or CPU)
			attention_mask = torch.ones_like(inputs)	# Attention mask (ensure valid token positions)

			# Generate predictions for the entire batch
			generated_outputs = model.generate(inputs, attention_mask=attention_mask, max_length=400)

			# Convert batch outputs to a list and append to results
			batch_results = generated_outputs.cpu().numpy().tolist()
			results.extend(batch_results)	# Extend the results list with batch results

	# Save the results to a file
	output_file: str = "reconstructed_results.txt"	# File to save the output sequences
	with open(output_file, "w") as f:
		# Write each sequence to the file
		for seq in results:
			f.write(" ".join(map(str, seq)) + "\n")

	print(f"Reconstructed results saved to {output_file}")	# Confirmation message

	# predicted_images = [pixel_to_image(sequence, colormap) for sequence in results]
	# show_images(predicted_images)