
import argparse
import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Local Qwen2.5-7B checkpoint directory"
    )

    parser.add_argument(
        "--output",
        default="reference_outputs.npz"
    )

    parser.add_argument(
        "--prompt",
        default="Explain how a transformer works."
    )

    args = parser.parse_args()

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    print("Loading Qwen in FP32...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager"
    )

    model = model.to("cpu")
    model.eval()

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]

    print("Input tokens:", input_ids.shape[1])

    if input_ids.shape[1] > 4096:
        raise ValueError("Prompt exceeds runtime context")

    reference = {
        "input_ids": input_ids.numpy()
    }

    hooks = []

    def capture(name):
        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            reference[name] = (
                output.detach()
                .to(torch.float32)
                .cpu()
                .numpy()
            )

        return hook

    # Embedding output.

    hooks.append(
        model.model.embed_tokens.register_forward_hook(
            capture("embedding")
        )
    )

    # Every transformer block output.

    for index, layer in enumerate(model.model.layers):

        hooks.append(
            layer.register_forward_hook(
                capture(f"layer_{index:02d}")
            )
        )

    # Final normalization.

    hooks.append(
        model.model.norm.register_forward_hook(
            capture("final_norm")
        )
    )

    # Vocabulary logits.

    hooks.append(
        model.lm_head.register_forward_hook(
            capture("logits")
        )
    )

    print("Running reference inference...")

    with torch.inference_mode():

        model(
            input_ids=input_ids,
            use_cache=False
        )

    for hook in hooks:
        hook.remove()

    print("Exporting reference tensors...")

    np.savez(
        args.output,
        **reference
    )

    metadata = {
        "model": args.model,
        "dtype": "float32",
        "sequence_length": int(input_ids.shape[1]),
        "num_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size
    }

    with open(
        args.output + ".json",
        "w"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("Reference export complete.")

    print("Saved:", args.output)

    for name, value in reference.items():

        print(
            name,
            value.shape,
            value.dtype
        )


if __name__ == "__main__":
    main()
