import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask
from dataclasses import dataclass, field, asdict
import tqdm

@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "The name of the model to use from the Hugging Face Hub."})
    dataset_name: str = field(default="TIGER-Lab/MMLU-Pro", metadata={"help": "The name of the dataset to use."})
    dataset_config: str = field(default="all", metadata={"help": "The configuration of the dataset."})
    split: str = field(default="test", metadata={"help": "The dataset split to use."})
    num_samples: int = field(default=0, metadata={"help": "The number of samples to evaluate. 0 to use all."})
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu", metadata={"help": "The device to run the model on ('cuda' or 'cpu')."})
    system_prompt: str = field(default='You are a helpful assistant. Start your answer with a single letter without any additional formatting.', metadata={"help": "The system prompt to use."})

@dataclass
class LoopingArguments:
    enable_loop: bool = field(default=False, metadata={"help": "Enable the looping experiment."})
    t_layer: int = field(default=0, metadata={"help": "The starting layer index for the loop."})
    k_layers: int = field(default=27, metadata={"help": "The number of layers in the loop."})
    num_loops: int = field(default=1, metadata={"help": "The number of times to loop."})

@dataclass
class HiddenStateArguments:
    enable_hidden_state_injection: bool = field(default=True, metadata={"help": "Enable the hidden state injection experiment."})
    num_iterations: int = field(default=2, metadata={"help": "Number of iterations for hidden state injection."})
    baseline_no_injection: bool = field(default=False, metadata={"help": "Run hidden state injection baseline without hidden state injection."})
    guess_prompt_template: str = field(default='{guess} would be my first guess. After some thinking, the answer is: \\boxed{{', metadata={"help": "The template for the guess prompt."})



def generate_with_loop(model, inputs, t, k, num_loops):
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # From Qwen3Model.forward
    use_cache = False
    past_key_values = None

    inputs_embeds = model.model.embed_tokens(input_ids)

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)

    # Prepare mask arguments
    mask_kwargs = {
        "config": model.model.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    # Create the masks
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }
    if "sliding_attention" in model.model.config.layer_types:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    # Layers 0 to t-1
    for i in range(t):
        decoder_layer = model.model.layers[i]
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs

    # This is the input to layer t
    h_t_in = hidden_states

    # This is the part we will loop over for the last token
    recurrent_hs_last_token = h_t_in[:, -1:, :].clone()

    # The loop
    for _ in range(num_loops):
        # Create input for the block of layers
        block_input = h_t_in.clone()
        block_input[:, -1:, :] = recurrent_hs_last_token

        h_temp = block_input
        for j in range(k):
            layer_idx = t + j
            if layer_idx >= len(model.model.layers):
                raise ValueError(f"t+k ({t + j}) is out of bounds. Number of layers: {len(model.model.layers)}")

            decoder_layer = model.model.layers[layer_idx]
            layer_outputs = decoder_layer(
                h_temp,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            h_temp = layer_outputs

        # Update the recurrent hidden state for the last token
        recurrent_hs_last_token = h_temp[:, -1:, :]

    # After the loop, we have the final hidden state for the last token at the input of layer t.
    # We create the final input for layer t.
    final_h_t_in = h_t_in.clone()
    final_h_t_in[:, -1:, :] = recurrent_hs_last_token

    hidden_states = final_h_t_in

    # Layers t to end
    for i in range(t, len(model.model.layers)):
        decoder_layer = model.model.layers[i]
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs

    # Final norm and lm_head
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    last_token_logits = logits[:, -1, :]
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)

    return torch.cat([input_ids, predicted_token_id.unsqueeze(-1)], dim=-1)


def generate_with_hidden_state_injection(model, tokenizer, inputs, text, num_iterations, device,
                                         baseline_no_injection=False, guess_prompt_template=None):
    special_token_str = "<|placeholder|>"
    special_token_id = tokenizer.convert_tokens_to_ids(special_token_str)

    def _prepare_inputs_for_iteration(guess_str_for_template):
        new_text = text + guess_prompt_template.format(guess=guess_str_for_template)

        new_inputs = tokenizer(new_text, return_tensors="pt")
        if device == "cuda":
            new_inputs = {k: v.to("cuda") for k, v in new_inputs.items()}

        input_ids = new_inputs['input_ids']

        special_token_indices = (input_ids[0] == special_token_id).nonzero(as_tuple=True)[0]
        if len(special_token_indices) == 0:
            raise ValueError(f"Special token '{special_token_str}' not found in the new prompt.")
        special_token_index = special_token_indices[0].item()
        return new_inputs, special_token_index

    if baseline_no_injection:
        with torch.no_grad():
            initial_outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        predicted_token = initial_outputs[0, -1:]

        for _ in range(num_iterations):
            new_inputs, special_token_index = _prepare_inputs_for_iteration(special_token_str)
            input_ids = new_inputs['input_ids']
            # print(f'Predicted token:{tokenizer.convert_ids_to_tokens(predicted_token)} {predicted_token}')
            input_ids[0, special_token_index] = predicted_token

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            predicted_token = torch.argmax(last_token_logits, dim=-1)

        return torch.cat([input_ids, predicted_token.unsqueeze(-1)], dim=-1)

    # Hidden state injection logic
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    final_token_hidden_states = [h[:, -1:, :].clone() for h in outputs.hidden_states]

    for _ in range(num_iterations):
        new_inputs, special_token_index = _prepare_inputs_for_iteration(special_token_str)
        input_ids = new_inputs['input_ids']

        # 3. Forward pass with hidden state injection
        use_cache = False
        past_key_values = None

        inputs_embeds = model.model.embed_tokens(input_ids)

        inputs_embeds[:, special_token_index:special_token_index + 1, :] = final_token_hidden_states[0]

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0)

        attention_mask = new_inputs['attention_mask']

        mask_kwargs = {
            "config": model.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if "sliding_attention" in model.model.config.layer_types:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        layer_hidden_states = [hidden_states.clone()]

        for i in range(len(model.model.layers)):
            decoder_layer = model.model.layers[i]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            if i + 1 < len(final_token_hidden_states):
                hidden_states[:, special_token_index:special_token_index + 1, :] = final_token_hidden_states[i + 1]

            layer_hidden_states.append(hidden_states.clone())

        final_token_hidden_states = [h[:, -1:, :] for h in layer_hidden_states]

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    last_token_logits = logits[:, -1, :]
    predicted_token_id = torch.argmax(last_token_logits, dim=-1)
    special_token = torch.argmax(logits[:, special_token_index, :], dim=-1)
    input_ids[:, special_token_index] = special_token

    return torch.cat([input_ids, predicted_token_id.unsqueeze(-1)], dim=-1)


def main():
    parser = HfArgumentParser((ScriptArguments, LoopingArguments, HiddenStateArguments))
    args: ScriptArguments
    loop_args: LoopingArguments
    hs_args: HiddenStateArguments
    args, loop_args, hs_args = parser.parse_args_into_dataclasses()

    wandb.init(
        project="recursive-LLM",
        config={
            "script_args": asdict(args),
            "loop_args": asdict(loop_args),
            "hs_args": asdict(hs_args),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if hs_args.enable_hidden_state_injection:
        special_token_str = "<|placeholder|>"
        tokenizer.add_special_tokens({'additional_special_tokens': [special_token_str]})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if hs_args.enable_hidden_state_injection:
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    model.to(args.device)

    correct = 0
    total = 0
    invalid_answers = 0

    if args.num_samples != 0:
        dataset = dataset.take(args.num_samples)

    dataset_name_lower = args.dataset_name.lower()

    pbar = tqdm.tqdm(dataset, desc='Prompt')
    for item in pbar:
        if "mmlu-pro" in dataset_name_lower:
            question_template = item["question"]
            options_list = item["options"]
            question = question_template
            question_template += '\n'


            labels = [chr(ord('A') + i) for i in range(len(options_list))]
            question += '\n'
            for i, option in enumerate(options_list):
                question += labels[i] + ' ' + option + '\n'
            correct_option_index = item['answer_index']
            answer_key = labels[correct_option_index]

            choices = {'label': labels, 'text': options_list}

        elif "mmlu" in dataset_name_lower:
            question_text = item["question"]
            choices_list = item["choices"]
            answer_idx = item["answer"]

            labels = [chr(ord('A') + i) for i in range(len(choices_list))]
            answer_key = labels[answer_idx]

            question = question_text
            for i, choice_text in enumerate(choices_list):
                question += f"\n{labels[i]}. {choice_text}"

            choices = {'label': labels, 'text': choices_list}
        else:  # ARC logic
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            for i, choice in enumerate(choices['text']):
                label = choices['label'][i]
                question += f"\n{label}. {choice}"

        messages= [
            {
                'content': args.system_prompt,
                'role': 'system',
            },
            {
                'content': question,
                'role': 'user',
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(text, return_tensors="pt")
        if args.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        if loop_args.enable_loop:
            outputs = generate_with_loop(model, inputs, loop_args.t_layer, loop_args.k_layers, loop_args.num_loops)
        elif hs_args.enable_hidden_state_injection:
            outputs = generate_with_hidden_state_injection(model, tokenizer, inputs, text, hs_args.num_iterations,
                                                         args.device, hs_args.baseline_no_injection,
                                                           hs_args.guess_prompt_template)
        else:
            outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if total == 0:
            print('Sample text:\n\n', generated_text)

        predicted_answer = generated_text[-1]

        if predicted_answer not in choices['label']:
            invalid_answers += 1
            print(f"Invalid answer. Correct answer: {answer_key}, labels: {choices['label']}, predicted answer: {predicted_answer}")

        if answer_key.lower() == predicted_answer.lower():
            correct += 1
        total += 1
        pbar.set_postfix_str(f"Accuracy: {correct / total:.2f}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    invalid_percentage = (invalid_answers / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Invalid answers: {invalid_percentage:.2f}% ({invalid_answers}/{total})")

    wandb.log({
        "accuracy": accuracy,
        "invalid_answer_percentage": invalid_percentage,
    })
    wandb.finish()

if __name__ == "__main__":
    main()
