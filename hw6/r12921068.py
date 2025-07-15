# Import Packages
from transformers import (
    AutoModelForCausalLM, # imports the model for causal language modeling
    AutoTokenizer, # imports the tokenizer for the model
    BitsAndBytesConfig, # imports the configuration for using bitsandbytes
    pipeline # imports the pipeline for text generation
)
from peft import (
    LoraConfig, # imports the configuration for LoRA
    get_peft_model, # imports the function to get the PEFT model
    PeftModel # imports the PEFT model
)
import os
import json
import torch
import random
import csv
from datasets import Dataset # Imports the Dataset class from the datasets library
from transformers.pipelines import Pipeline
from trl import SFTConfig, SFTTrainer # Imports the SFTConfig and SFTTrainer classes from the trl library
from tqdm import tqdm # Imports the tqdm library for progress bars

def load_jsonlines(file_name: str):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str, answer: any, mode: str) -> dict: # Function to create n-shot chats
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    # TODO: Use fixed few-shot examples
    for qna in nshot_data[:n]: # Samples n examples from the n-shot data
        chats.append(
            {
                'role': 'user',
                'content': f'Q: {qna["question"]}' # Creates a user message with the question
            }
        )
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {qna["answer"]}' # Creates an assistant message with the answer
            }
        )

    chats.append(
        {
            'role': 'user',
            'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.' # Creates a user message with the question and instructions
        }
    )
    if mode == 'train':
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {answer}' # Creates an assistant message with the answer
            }
        )

    return chats # Returns the list of chats

def get_response(chats: list, generator: Pipeline): # Function to get the response from the model
    gen_text = generator(chats)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content'] # Returns the content of the last generated text

def extract_ans_from_response(answer: str): # Function to extract the answer from the response
    answer = answer.split('####')[-1].strip() # Splits the answer by '####' and takes the last part

    for remove_char in [',', '$', '%', 'g']: # Removes unwanted characters from the answer
        answer = answer.replace(remove_char, '')

    return answer # Returns the extracted answer

def load_csv(file_name: str):
    csvfile = open(file_name)
    rows = csv.DictReader(csvfile)
    questions = []
    for row in rows:
        questions.append(row['prompt_text'])
    return questions

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Sets the CUDA device to use
    device = torch.device('cuda:0') # Creates a CUDA device object

    random.seed(42) # Sets the random seed for reproducibility


    # LLM Fine-tuning
    # 1. Load Model & Tokenizer
    sft_model_name = 'meta-llama/Llama-3.2-1B-Instruct' # Specifies the name of the pre-trained model to use
    sft_bnb_config = BitsAndBytesConfig( # Configuration for using bitsandbytes
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    sft_model = AutoModelForCausalLM.from_pretrained( # Loads the pre-trained model
        pretrained_model_name_or_path=sft_model_name,
        quantization_config=sft_bnb_config,
        low_cpu_mem_usage=True,
    )
    sft_tokenizer = AutoTokenizer.from_pretrained( # Loads the tokenizer for the model
        pretrained_model_name_or_path=sft_model_name,
    )
    sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Adds a special token for padding
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # TODO: Adds dropout
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    peft_model = get_peft_model(sft_model, peft_config)

    # 2. Format GSM8K Data for Fine-tuning
    gsm8k_train = load_jsonlines('gsm8k_train_self-instruct.jsonl') # You can use refined gsm8k_train_self-instruct.jsonl for fine-tuning
    gsm8k_eval = load_jsonlines('gsm8k_eval_self-instruct.jsonl') # You can use refined gsm8k_train_self-instruct.jsonl for fine-tuning

    formatted_gsm8k = []
    TRAIN_N_SHOT = 8 # TODO: Give model more examples
    max_token_len = 0 # Record token length of dataset and prevent data from truncation
    for qna in gsm8k_train: # Iterates over the GSM8K training data
        chats = nshot_chats(nshot_data=gsm8k_eval, n=TRAIN_N_SHOT, question=qna['question'], answer=qna['answer'], mode='train') # Creates n-shot chats for the current example
        train_sample = sft_tokenizer.apply_chat_template(chats, tokenize=False) # Applies the chat template to the chats
        train_sample = train_sample[train_sample.index("<|eot_id|>") + len("<|eot_id|>"):] # Remove Cutting Knowledge Date in prompt template
        formatted_gsm8k.append( # Appends the formatted example to the list
            {
                'text': train_sample # Adds the text of the example
            }
        )
        max_token_len = max(max_token_len, len(sft_tokenizer(train_sample)['input_ids'])) # Updates the maximum token length

    formatted_gsm8k = Dataset.from_list(formatted_gsm8k) # Creates a dataset from the list of formatted examples
    
    
    # 3. Fine-tuning
    # trainer
    training_arguments = SFTConfig( # Configuration for the SFT trainer
        seed=1126,
        data_seed=1126,
        output_dir=f"sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=10, # 5: If you use fixed few-shot examples, increase epoch
        logging_strategy="steps",
        logging_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        lr_scheduler_type='linear',
        learning_rate=1e-5, # TODO: Decrease learning rate
        # TODO: Add weight decay
        weight_decay=1e-4,
        bf16=True,
        group_by_length=True,
        max_seq_length=max_token_len,
        dataset_text_field='text',
        report_to='none',
    )
    trainer = SFTTrainer( # Creates the SFT trainer
        model=peft_model,
        train_dataset=formatted_gsm8k,
        peft_config=peft_config,
        processing_class=sft_tokenizer,
        args=training_arguments,
    )
    trainer.train() # Starts the training process

    # LLM Inference
    # 1. Load Adapter Checkpoint
    generator = pipeline( # Creates a text generation pipeline
        'text-generation',
        model=sft_model,
        tokenizer=sft_tokenizer,
        pad_token_id=sft_tokenizer.eos_token_id,
        max_new_tokens=1024, # TODO: Increase max_new_tokens for longer output
        # TODO: Use greedy decoding strategy
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
    )
    adapter_path = 'sft/checkpoint-18680' # TODO: Evaluate different checkpoints
    pipeline.model = PeftModel.from_pretrained( # Loads the adapter checkpoint
        sft_model,
        adapter_path
    )
    # 2. GSM8K
    gsm8k_predictions = []
    TEST_N_SHOT = 8 # TODO: give model more examples

    gsm8k_test_public = load_jsonlines('gsm8k_test_public.jsonl') # Loads the GSM8K public test data
    gsm8k_total = len(gsm8k_test_public) # Gets the total number of examples in the public test data
    gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Public Test Data Evaluation', postfix='Current Accuracy = 0.000') # Creates a progress bar for the public test data evaluation

    correct = 0

    for i, qna in enumerate(gsm8k_test_public): # Iterates over the public test data

        messages = nshot_chats(nshot_data=gsm8k_eval, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
        response = get_response(messages, generator) # Gets the response from the model

        pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
        true_ans = extract_ans_from_response(qna["answer"]) # Extracts the true answer from the example
        if pred_ans == true_ans: # Checks if the predicted answer is correct
            correct += 1 # Increments the correct count if the prediction is correct
        gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

        gsm8k_progress_bar.set_postfix_str(f'Current Accuracy = {correct/(i+1):.3f}') # Updates the progress bar with the current accuracy
        gsm8k_progress_bar.update() # Updates the progress bar

    gsm8k_progress_bar.close() # Closes the progress bar

    print(f'GSM8K Public Test Data Evaluation Complete, Total Accuracy: {correct/gsm8k_total:.3f}') # Prints the total accuracy on the public test data

    gsm8k_test_private = load_jsonlines('gsm8k_test_private.jsonl') # Loads the GSM8K private test data
    gsm8k_total = len(gsm8k_test_private) # Gets the total number of examples in the private test data
    gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Private Test Data Inference') # Creates a progress bar for the private test data evaluation

    for i, qna in enumerate(gsm8k_test_private): # Iterates over the private test data

        messages = nshot_chats(nshot_data=gsm8k_eval, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
        response = get_response(messages, generator) # Gets the response from the model

        pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
        gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

        gsm8k_progress_bar.update() # Updates the progress bar

    gsm8k_progress_bar.close() # Closes the progress bar

    print(f'GSM8K Private Test Data Inference Complete') # Prints a message indicating that the private test data evaluation is complete

    # 3. AILuminate
    ailuminate_predictions = []

    ailuminate_test = load_csv('ailuminate_test.csv') # Loads the AILuminate test data
    ailuminate_total = len(ailuminate_test) # Gets the total number of examples in the AILuminate test data
    ailuminate_progress_bar = tqdm(total=ailuminate_total, desc='AILuminate Test Data Evaluation') # Creates a progress bar for the AILuminate test data evaluation

    for i, question in enumerate(ailuminate_test): # Iterates over the AILuminate test data

        message = [
            {
                'role': 'user',
                'content': question
            }
        ]
        response = get_response(message, generator) # Gets the response from the model
        ailuminate_predictions.append(response) # Appends the response to the list of predictions

        ailuminate_progress_bar.update() # Updates the progress bar
    ailuminate_progress_bar.close() # Closes the progress bar

    print(f'AIluminate Test Data Evaluation Complete')

    # Create Submission File
    # Combine the results into one file.
    STUDENT_ID = 'r12921068' # TODO: Add your student id
    with open(f'./{STUDENT_ID}.txt', 'w') as output_f:
        print(gsm8k_predictions + ailuminate_predictions, file=output_f) # Prints the predictions to the output file

if __name__ == "__main__":
    main()