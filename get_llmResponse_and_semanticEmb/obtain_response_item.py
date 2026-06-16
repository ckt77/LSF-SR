import pickle
import json
import time
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datetime import datetime
import traceback

# put your Hugging Face token in hf_token.txt
_hf_token_file = Path(__file__).resolve().parent / "hf_token.txt"
with open(_hf_token_file, encoding="utf-8") as f:
    hf_token = f.read().strip()

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate item summary responses using LLM')
parser.add_argument('--start_item', type=int, default=None,
                    help='Start item ID (inclusive). If not specified, processes all items.')
parser.add_argument('--end_item', type=int, default=None,
                    help='End item ID (inclusive). If not specified, processes all items.')
parser.add_argument('--test_first', action='store_true',
                    help='Test with the first item only (for testing purposes)')
args = parser.parse_args()

# indicate the dataset
file_path = '../build_datasets_and_prompts/data/Beauty/'

# input request file
input_path = file_path + "item_prompt_input.pkl"

# Generate output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.start_item is not None and args.end_item is not None:
    output_filename = f'responses_item_summary_{args.start_item}_{args.end_item}_{timestamp}.json'
else:
    output_filename = f'responses_item_summary_{timestamp}.json'

write_path = file_path + output_filename
print(f"Output file: {write_path}")

with open(input_path, 'rb') as pickle_file:
    question_dic = pickle.load(pickle_file)

# System input for item analysis
system_input = """Assume you are a beauty and personal care recommendation expert. Please help me analyze a specific beauty product."""

# Model configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Loading model {model_name} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": device},
    token=hf_token
)
model.eval()

# Set generation config to ensure deterministic output
generation_config = GenerationConfig(
    do_sample=False,
    temperature=None,
    top_p=None,
    max_new_tokens=300
)
model.generation_config = generation_config


def generate_response(item_prompt_content):
    """Generate response using local llama3-8b model"""
    user_content = item_prompt_content

    if hasattr(tokenizer,
               'chat_template') and tokenizer.chat_template is not None:
        try:
            messages = [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_content}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        except Exception as e:
            print(
                f"Warning: apply_chat_template failed: {e}, using manual format")

            # Fallback to manual format
            prompt = (
                f"{system_input}\n\n{item_prompt_content}\n\n"
                "Please provide your answer in JSON format:"
            )

    else:
        prompt = (
            f"{system_input}\n\n{item_prompt_content}\n\n"
            "Please provide your answer in JSON format:"
        )

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096).to(device)

    # Generate response with deterministic settings (greedy decoding)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode response (remove the input prompt part)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def main():
    start_time = time.time()
    responses = {}

    # Get all item requests
    all_item_requests = list(question_dic.items())

    # If test mode, only process the first item
    if args.test_first:
        print("=" * 80)
        print("TEST MODE: Processing only the first item")
        print("=" * 80)
        item_requests = all_item_requests[:1]
        print(f"Testing with item key: {item_requests[0][0]}")
        print(
            f"Item prompt preview (first 500 chars):\n{item_requests[0][1][:500]}...")
        print("=" * 80)

    else:
        if args.start_item is not None and args.end_item is not None:
            item_requests = []
            for key, content in all_item_requests:
                try:
                    item_id = int(key)
                    if args.start_item <= item_id <= args.end_item:
                        item_requests.append((key, content))

                except (ValueError, TypeError):
                    continue

            if len(item_requests) == 0:
                print(
                    f"No items found for item ID range {args.start_item} to {args.end_item}!")
                return responses

            else:
                print(
                    f"Processing items {args.start_item} to {args.end_item} (inclusive)")
                print(f"Found {len(item_requests)} items in this range")

        elif args.start_item is not None:
            item_requests = []
            for key, content in all_item_requests:
                try:
                    item_id = int(key)
                    if item_id >= args.start_item:
                        item_requests.append((key, content))

                except (ValueError, TypeError):
                    continue

            if len(item_requests) == 0:
                print(f"No items found for item ID >= {args.start_item}!")
                return responses

            else:
                print(f"Processing items from {args.start_item} onwards")
                print(f"Found {len(item_requests)} items")

        elif args.end_item is not None:
            item_requests = []
            for key, content in all_item_requests:
                try:
                    item_id = int(key)
                    if item_id <= args.end_item:
                        item_requests.append((key, content))

                except (ValueError, TypeError):
                    continue

            if len(item_requests) == 0:
                print(f"No items found for item ID <= {args.end_item}!")
                return responses

            else:
                print(f"Processing items up to {args.end_item}")
                print(f"Found {len(item_requests)} items")

        else:
            item_requests = all_item_requests
            print(f"Processing all items")
            print(f"Found {len(item_requests)} items")

    # Process each item with periodic saving
    save_interval = 20  # Save every 20 items
    for idx, (key, content) in enumerate(item_requests, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"\nProcessing item {idx}/{len(item_requests)}: {key}")

        try:
            response = generate_response(content)
            responses[key] = response

        except Exception as e:
            print(f"Error processing {key}: {e}")
            traceback.print_exc()
            responses[key] = None

        # Periodic save every save_interval items
        if idx % save_interval == 0:
            with open(write_path, 'w') as f:
                json.dump(responses, f, indent=4)
            print(
                f"Progress saved: {idx}/{len(item_requests)} items processed ({len(responses)} responses saved)")

    # Final save to ensure all results are written
    with open(write_path, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"Final save completed: All {len(responses)} responses saved")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
    print(f"Processed {len(responses)} item(s)")
    print(f"Results saved to: {write_path}")

    return responses


if __name__ == "__main__":
    responses = main()
    print(".......", len(list(responses.keys())), ".......")
