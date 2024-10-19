import os
import argparse
import vllm
import json
import torch
import random
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for tagging")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="allenai/Llama-3-8B-Instruct-Analyzer",
        help="Tagger model name on huggingface hub or a local path"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Tokenizer name on huggingface hub or a local path"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="Huggingface dataset name or a path to the local data file (in .jsonl format)"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The dataset split if necessary"
    )
    parser.add_argument(
        "--hf_data_files",
        type=str,
        nargs="*",
        help="Argument to pass to the dataset function `load_dataset` as `data_files`"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/tagging_results.jsonl",
        help="Path to save the output file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max number of samples to process from the dataset. If not specified, process all samples."
    )
    
    # Dimensions to analyze
    parser.add_argument(
        "--tagging_dimensions",
        type=str,
        nargs="*",
        help="List of dimensions to tag. If not specified, we will use all available dimensions"
    )
    parser.add_argument(
        "--tagging_instructions",
        type=str,
        default="src/tagging_configs.json",
        help="A file that contains the tagging instructions of each dimension"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--stop_sequences",
        nargs='*',
        default=None,
        help="List of sequences to stop generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for generation. If not specified, vllm will auto batchify, but the results will be saved only after the entire dataset is generated."
    )
    args = parser.parse_args()
    return args


def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def format_raw_query(instance):
    if "prompt" in instance:
        messages = [{"role": "user", "content": instance["prompt"]}]
    elif "text" in instance:
        messages = [{"role": "user", "content": instance["text"]}]
    elif "messages" in instance:
        messages = instance["messages"]
    elif "chosen" in instance and "rejected" in instance:
        messages = instance["chosen"][:-1]
    else:
        raise ValueError("Invalid data format. Expected 'prompt' or 'messages' in the instance.")

    query_messages = []
    for idx, message in enumerate(messages):
        # We allow system messages to be included in the query.
        # We stop by the second user query to avoid considering a long conversation history.
        # This means we use the meta tag of at most the second request to represent the entire conversation.
        # We also don't just use the first user query if there are more, 
        # because it might be a greeting or a short message.
        if message["role"] == "system":
            query_messages.append({"role": "system", "content": message["content"]})
        elif message["role"] == "user":
            # at most 2 user queries
            if sum([1 for m in query_messages[:idx] if m["role"] == "user"]) <= 2:
                query_messages.append({"role": "user", "content": message["content"]})
            else:
                break
        elif message["role"] == "assistant":
            # at most 1 assistant response
            # only include the assistant response if there are at least 2 user queries 
            # and there is no assistant response yet
            if sum([1 for m in messages if m["role"] == "user"]) >= 2 \
                and sum([1 for m in query_messages if m["role"] == "assistant"]) == 0:
                query_messages.append({"role": "assistant", "content": message["content"]})
        else:
            raise ValueError(f"Invalid role: {message['role']}")
                
    raw_query = ""
    for m in messages:
        raw_query += f"<|{m['role']}|>\n{m['content']}\n"
    return raw_query


def main():
    args = parse_args()
    print("Parsing arguments...")
    print(f"Parsed arguments: {args}")

    print("Initializing VLLM model...")
    # Initialize the VLLM model
    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path if args.model_name_or_path else args.model_name_or_path,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
    )
    
    # Add more logging for later stages
    print("Loading tagging instructions...")
    # Load tagging instructions
    with open(args.tagging_instructions, 'r') as f:
        tagging_instructions = json.load(f)
    
    # Create a dictionary of dimensions and their instructions
    tagging_instructions = {it["dimension"]: it["tagging_instruction"] for it in tagging_instructions}
    if args.tagging_dimensions:
        # Validate that all requested dimensions have instructions
        missing_dimensions = set(args.tagging_dimensions) - set(tagging_instructions.keys())
        if missing_dimensions:
            raise ValueError(f"Missing instructions for dimensions: {', '.join(missing_dimensions)}")
        tagging_instructions = {dim: tagging_instructions[dim] for dim in args.tagging_dimensions}
    tagging_dimensions = list(tagging_instructions.keys())

    # Load the data we want to tag
    print("Loading dataset...")
    if args.dataset_name_or_path.endswith(".json") or args.dataset_name_or_path.endswith(".jsonl"):
        dataset_to_tag = load_dataset("json", data_files=args.dataset_name_or_path, split=args.dataset_split)
    else:
        # Try loading from huggingface hub
        dataset_to_tag = load_dataset(args.dataset_name_or_path, data_files=args.hf_data_files, split=args.dataset_split)
        
    if args.max_samples:
        # Randomly select a subset of the dataset if max_samples is specified
        if args.max_samples < len(dataset_to_tag):
            dataset_to_tag = dataset_to_tag.shuffle(seed=42).select(range(args.max_samples))
    print(f"Dataset loaded. Number of samples: {len(dataset_to_tag)}")

    # Prepare the tagging prompts
    print("Preparing tagging prompts...")
    tagging_prompts = []
    tokenizer = model.get_tokenizer()
    for instance in dataset_to_tag:
        raw_query = format_raw_query(instance)
        for tagging_dimension, tagging_instruction in tagging_instructions.items():
            tagging_prompt = tagging_instruction + "\n\nData:\n" + "[START]\n" + raw_query + "[END]\n"
            tagging_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": tagging_prompt}], 
                add_generation_prompt=True, 
                tokenize=False
            )
            tagging_prompts.append(tagging_prompt)

    # print the first 10 tagging prompts
    print("Random 10 tagging prompts:")
    for i in range(10):
        # random sample an index
        index = random.randint(0, len(tagging_prompts) - 1)
        print(f"===== Random tagging prompt {i}, index: {index} =====")
        print(tagging_prompts[index])
        
    # Generate tags using the VLLM model
    print("Generating tags using VLLM model...")
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop_sequences
    )
    
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Generate tags using the VLLM model in batches
    print("Generating tags using VLLM model...")
    batch_size = args.batch_size if args.batch_size else len(tagging_prompts)
    results = []
    batch_count = 0

    # Open the main output file
    with open(args.output_path, 'w') as output_file:
        for i in range(0, len(tagging_prompts), batch_size):
            batch_prompts = tagging_prompts[i:i+batch_size]
            outputs = model.generate(batch_prompts, sampling_params)

            # Process the outputs for this batch
            batch_results = []
            for j, output in enumerate(outputs):
                global_index = i + j
                instance_index = global_index // len(tagging_dimensions)
                dimension_index = global_index % len(tagging_dimensions)
                
                if dimension_index == 0:
                    batch_results.append(dataset_to_tag[instance_index].copy())
                    batch_results[-1]["tags"] = {}
                
                dimension = tagging_dimensions[dimension_index]
                generated_text = output.outputs[0].text.strip()
                # try parsing the generated result as json
                try:
                    parsed_json = json.loads(generated_text)
                    if dimension in parsed_json:
                        batch_results[-1]["tags"][dimension] = parsed_json[dimension]
                    else:
                        batch_results[-1]["tags"][dimension] = None
                except json.JSONDecodeError:
                    batch_results[-1]["tags"][dimension] = None 

            # Save results for this batch to output_file
            for item in batch_results:
                json.dump(item, output_file, default=str)
                output_file.write("\n")
            
            results.extend(batch_results)
            batch_count += 1
            print(f"Processed batch {batch_count} ({len(results)} / {len(dataset_to_tag)} instances processed so far)...")

    print(f"Processing complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()

