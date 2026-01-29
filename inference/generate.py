import json
import os
from argparse import ArgumentParser
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def load_memory_store(store_path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(store_path):
        return {"personas": {}}
    with open(store_path) as f:
        data = json.load(f)
    personas = data.get("personas", {}) if isinstance(data, dict) else {}
    return {"personas": personas}


def save_memory_store(store_path: str, data: Dict[str, Dict[str, List[str]]]) -> None:
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(data, f, indent=2)


def build_system_message(persona_name: str, persona_data: Dict[str, List[str]]) -> Dict[str, str]:
    description = persona_data.get("description", "")
    memories = persona_data.get("memories", [])
    memory_lines = "\n".join(f"- {memory}" for memory in memories) or "- (none)"
    content = (
        f"You are using the persona: {persona_name}.\n"
        f"Persona description: {description or '(none provided)'}.\n"
        "Persistent memories:\n"
        f"{memory_lines}"
    )
    return {"role": "system", "content": content}


def parse_persona_command(prompt: str) -> List[str]:
    parts = prompt.strip().split(maxsplit=2)
    if len(parts) == 1:
        return [parts[0]]
    if len(parts) == 2:
        return parts
    return parts


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    device = next(model.parameters()).device
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.as_tensor(t, dtype=torch.long, device=device)
    prev_pos = 0
    finished = torch.zeros(len(prompt_tokens), device=device, dtype=torch.bool)
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    memory_store: str = "",
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])

    if interactive:
        messages = []
        store_path = memory_store or os.path.join(os.path.expanduser("~"), ".deepseek", "memory_profiles.json")
        memory_store_data = load_memory_store(store_path)
        active_persona: Optional[str] = None
        print("\nPersona & memory commands: /persona list|create|use|delete|show, /memory add|list|clear, /clear, /exit")
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            elif prompt.startswith("/persona"):
                parts = parse_persona_command(prompt)
                if len(parts) == 1:
                    print("Usage: /persona list|create|use|delete|show")
                    continue
                command = parts[1]
                personas = memory_store_data["personas"]
                if command == "list":
                    if not personas:
                        print("No personas saved.")
                    else:
                        for name in sorted(personas.keys()):
                            marker = "*" if name == active_persona else " "
                            print(f"{marker} {name}")
                elif command == "create":
                    if len(parts) < 3:
                        print("Usage: /persona create <name> <description>")
                        continue
                    name_and_desc = parts[2].split(maxsplit=1)
                    if len(name_and_desc) < 2:
                        print("Usage: /persona create <name> <description>")
                        continue
                    name, description = name_and_desc
                    if name in personas:
                        print(f"Persona '{name}' already exists.")
                        continue
                    personas[name] = {"description": description, "memories": []}
                    save_memory_store(store_path, memory_store_data)
                    active_persona = name
                    print(f"Created persona '{name}' and set as active.")
                elif command == "use":
                    if len(parts) < 3:
                        print("Usage: /persona use <name>")
                        continue
                    name = parts[2]
                    if name not in personas:
                        print(f"Persona '{name}' not found.")
                        continue
                    active_persona = name
                    print(f"Active persona set to '{name}'.")
                elif command == "delete":
                    if len(parts) < 3:
                        print("Usage: /persona delete <name>")
                        continue
                    name = parts[2]
                    if name not in personas:
                        print(f"Persona '{name}' not found.")
                        continue
                    personas.pop(name)
                    if active_persona == name:
                        active_persona = None
                    save_memory_store(store_path, memory_store_data)
                    print(f"Deleted persona '{name}'.")
                elif command == "show":
                    name = active_persona if len(parts) < 3 else parts[2]
                    if not name:
                        print("No active persona. Use /persona use <name>.")
                        continue
                    if name not in personas:
                        print(f"Persona '{name}' not found.")
                        continue
                    persona_data = personas[name]
                    print(f"Persona: {name}")
                    print(f"Description: {persona_data.get('description', '')}")
                    memories = persona_data.get("memories", [])
                    if memories:
                        print("Memories:")
                        for memory in memories:
                            print(f"- {memory}")
                    else:
                        print("Memories: (none)")
                else:
                    print("Unknown persona command. Use /persona list|create|use|delete|show")
                continue
            elif prompt.startswith("/memory"):
                parts = prompt.strip().split(maxsplit=2)
                if len(parts) == 1:
                    print("Usage: /memory add|list|clear")
                    continue
                command = parts[1]
                if not active_persona:
                    print("No active persona. Use /persona use <name>.")
                    continue
                persona_data = memory_store_data["personas"].get(active_persona)
                if not persona_data:
                    print(f"Persona '{active_persona}' not found.")
                    continue
                if command == "add":
                    if len(parts) < 3:
                        print("Usage: /memory add <text>")
                        continue
                    persona_data.setdefault("memories", []).append(parts[2])
                    save_memory_store(store_path, memory_store_data)
                    print("Memory added.")
                elif command == "list":
                    memories = persona_data.get("memories", [])
                    if memories:
                        for memory in memories:
                            print(f"- {memory}")
                    else:
                        print("No memories saved.")
                elif command == "clear":
                    persona_data["memories"] = []
                    save_memory_store(store_path, memory_store_data)
                    print("Memories cleared.")
                else:
                    print("Unknown memory command. Use /memory add|list|clear")
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_messages = list(messages)
            if active_persona:
                persona_data = memory_store_data["personas"].get(active_persona, {})
                prompt_messages = [build_system_message(active_persona, persona_data)] + prompt_messages
            prompt_tokens = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--memory-store", type=str, default="", help="Path to persona/memory JSON store")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature,
        args.memory_store,
    )
