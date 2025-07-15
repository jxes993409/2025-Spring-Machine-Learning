from datasets import Dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported
import json

def data_formulate(data, tokenizer):
	messages = [
		{"role": "system", "content": "Your entire response must be 100 characters or less."},
		{"role": "user", "content": data['prompt']},
	]
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	return prompt

def extract_assistant_response(text):
	try:
		# Split by assistant header marker
		parts = text.split("<|start_header_id|>assistant<|end_header_id|>")
		if len(parts) < 2:
			return None

		# Split by end of text marker
		assistant_part = parts[1]
		response_parts = assistant_part.split("<|eot_id|>")

		# Clean up any whitespace
		return response_parts[0].strip()
	except Exception as e:
		print(f"Error extracting assistant response: {e}")
		return None

def main(
	num_epoch: int,
	data_size: int,
	support_ratio: int,
):
	max_seq_length = 512
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = "unsloth/llama-3-8b-Instruct",
		max_seq_length = max_seq_length,
		load_in_4bit = True,
	)

	with open("ML2025Spring-HW7/train.json", 'r') as jsonfile:
		full_data = json.load(jsonfile)

	with open("ML2025Spring-HW7/test.json", 'r') as jsonfile:
		test_data = json.load(jsonfile)


	original_model_response = []
	for data in test_data:
		id = data['id']
		prompt = data['prompt']
		# print(f'\nQuestion {id}: {prompt}')
		inputs = data_formulate(data, tokenizer)
		outputs = model.generate(
			**tokenizer(inputs, return_tensors = "pt").to("cuda"),
			max_new_tokens = 128,
			do_sample=False
		)
		output = tokenizer.batch_decode(outputs)[0]
		output = extract_assistant_response(output)
		original_model_response.append(output)
		# print()
		# print(output)

	#### DO NOT CHANGE ####

	# Select part of the data for training
	training_data = full_data[:data_size]

	# Define the size of the support dataset
	support_data_size = int(data_size * support_ratio)

	# Prepare the data for the training dataset
	prompt_list = [data_formulate(data, tokenizer) for data in training_data]
	chosen_list = [data['support'] for data in training_data[:support_data_size]] + [data['oppose'] for data in training_data[support_data_size:]]
	rejected_list = [data['oppose'] for data in training_data[:support_data_size]] + [data['support'] for data in training_data[support_data_size:]]

	# Create the training dataset
	train_dataset = Dataset.from_dict({'prompt': prompt_list, 'chosen': chosen_list, 'rejected': rejected_list})

	#### DO NOT CHANGE ####

	model = FastLanguageModel.get_peft_model(
		model,
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
						"gate_proj", "up_proj", "down_proj",],

		r = 16,           # Larger = higher accuracy, but might overfit
		lora_alpha = 16,  # Recommended alpha == r at least
		lora_dropout = 0.1,
		bias = "none",
		random_state = 3407, # Do not modify the random_state for reproducibility
		use_rslora = False,  # We support rank stabilized LoRA
		loftq_config = None, # And LoftQ
	)

	#### DO NOT CHANGE ####

	dpo_trainer = DPOTrainer(
		model = model,
		ref_model = None,
		args = DPOConfig(
			per_device_train_batch_size = 2,
			gradient_accumulation_steps = 4,
			warmup_ratio = 0.1,
			num_train_epochs = num_epoch,
			learning_rate = 1e-4,
			fp16 = not is_bfloat16_supported(),
			bf16 = is_bfloat16_supported(),
			logging_steps = 1,
			optim = "paged_adamw_8bit",
			weight_decay = 0.0,
			lr_scheduler_type = "linear",
			seed = 42,
			output_dir = "outputs",
			report_to = "none",
		),
		beta = 0.1,
		train_dataset = train_dataset,
		tokenizer = tokenizer,
	)

	dpo_trainer.train()

	aligned_model_response = []
	for data in test_data:
		id = data['id']
		prompt = data['prompt']
		# print(f'\nQuestion {id}: {prompt}')
		inputs = data_formulate(data, tokenizer)
		outputs = model.generate(
			**tokenizer(inputs, return_tensors = "pt").to("cuda"),
			max_new_tokens = 128,
			do_sample=False
		)
		output = tokenizer.batch_decode(outputs)[0]
		output = extract_assistant_response(output)
		aligned_model_response.append(output)
		# print()
		# print(output)

	student_id = "r12921068" # TODO: fill in your student id here.
	dir_name = "" # TODO: If you use machines other than colab, please adjust the directory here.
	# Do NOT change the following for this block.
	file_name = f"{student_id}_hw7_epoch{num_epoch}_ratio{support_ratio}_size{data_size}.json"
	output_list = []
	for data in test_data:
		original_response = original_model_response.pop(0)
		aligned_response = aligned_model_response.pop(0)
		output_list.append({"id": data["id"], "prompt": data["prompt"], "original_response": original_response, "aligned_response": aligned_response})
	output_data = {"num_epoch": num_epoch, "data_size": data_size, "support_ratio": support_ratio, "results": output_list}
	with open(file_name, "w") as output_file:
		json.dump(output_data, output_file, indent=4)

if __name__ == "__main__":
	main(
		num_epoch=3,
		data_size=10,
		support_ratio=0
	)