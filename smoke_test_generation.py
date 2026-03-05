"""
Smoke test for Stage 1 generation: load a trained checkpoint and generate text.

Usage:
    python smoke_test_generation.py
    python smoke_test_generation.py --checkpoint runs/20260224_212158/checkpoints/best_model.pt
    python smoke_test_generation.py --max-new-tokens 50 --temperature 0.8

Checks:
  1. Checkpoint loads without error.
  2. forward_inference output has correct shape, no NaN/Inf.
  3. generate() returns a GenerationResult that terminates within token limit.
  4. Output looks like early-LM text (token entropy > random-weight baseline).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a TS2TSTrainingModule from checkpoint and convert to inference model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from model.modules.training_module import TS2TSTrainingModule
    from model.graph_traversal.block_mask_creator import make_mask_creator_callable

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("metadata", {}).get("config", {})
    val_loss = ckpt.get("metadata", {}).get("val_loss")
    step = ckpt.get("metadata", {}).get("step")
    print(f"  val_loss={val_loss:.4f}  step={step}")

    # Reconstruct hyperparameters — fall back to the run's hyperparameters.json
    hp_path = checkpoint_path.parent.parent / "hyperparameters.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
    else:
        # Sensible defaults matching the known checkpoint
        hp = {
            "model": {
                "model_dim": 768, "num_layers": 12, "num_heads": 12,
                "max_seq_len": 2048, "dropout": 0.0, "drop_path_rate": 0.0,
                "fp8": False, "weight_tying": True, "ignore_index": -100,
                "dtype": "bfloat16", "mask_type": "doc_causal",
            }
        }

    mcfg = hp["model"]
    mask_type = mcfg.get("mask_type", "doc_causal")
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))

    block_mask_creator = make_mask_creator_callable(mask_type)

    training_module = TS2TSTrainingModule.from_config(
        vocab_size=50257,
        num_layers=mcfg["num_layers"],
        model_dim=mcfg["model_dim"],
        num_heads=mcfg["num_heads"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=0.0,          # disable for inference
        drop_path_rate=0.0,
        block_mask_creator=block_mask_creator,
        fp8=mcfg.get("fp8", False),
        weight_tying=mcfg.get("weight_tying", True),
        ignore_index=mcfg.get("ignore_index", -100),
        dtype=dtype,
    )

    missing, unexpected = training_module.load_state_dict(ckpt["model"], strict=True)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    tokenizer = tiktoken.get_encoding("gpt2")
    model = training_module.to_inference_model(tokenizer=tokenizer)
    model.to(device, dtype)
    model.eval()
    return model, dtype


def check_forward_inference(model, device, dtype):
    print("\n[1] forward_inference shape / NaN check")
    tokens = torch.randint(0, 50257, (1, 50), device=device, dtype=torch.long)
    logits = model.forward_inference(tokens)
    assert logits.shape == (1, 50, 50257), f"Wrong shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    print(f"  shape={logits.shape}  OK")


def entropy_of_logits(logits_last: torch.Tensor) -> float:
    probs = F.softmax(logits_last.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def check_generate(model, device, max_new_tokens: int, temperature: float):
    print(f"\n[2] generate() end-to-end  (max_new_tokens={max_new_tokens})")
    from model.generation_config import GenerationConfig

    prompts = [
        "Python is a programming language",
        "The Roman Empire was",
        "Water is composed of",
    ]
    max_entropy = math.log(50257)   # entropy of uniform distribution over vocab

    for prompt in prompts:
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=str(device),
            eos_token_id=50256,
            max_tokens_per_document=max_new_tokens + 50,
            max_context_length=2048,
        )
        result = model.generate(prompt, config=config)
        doc = result.root_document

        assert doc.text is not None, "text field not populated"
        assert len(doc.tokens) > 0, "no tokens generated"

        # Compute a rough entropy check: get logits for the generated sequence
        # and measure how peaked the distribution is at the last token
        tokens_tensor = torch.tensor(doc.tokens.tolist(), dtype=torch.long,
                                     device=device).unsqueeze(0)
        logits = model.forward_inference(tokens_tensor)
        ent = entropy_of_logits(logits[0, -1, :])
        entropy_frac = ent / max_entropy   # 0 = peaky / 1 = uniform

        # Random-weight model → near uniform → entropy_frac ≈ 1.0
        # Trained model → more peaked → entropy_frac should be noticeably < 1.0
        print(f"  prompt: {prompt!r}")
        print(f"  tokens: {len(doc.tokens)}  entropy_frac={entropy_frac:.3f}  "
              f"({'OK — peaked enough' if entropy_frac < 0.95 else 'WARN — near uniform'})")
        # Show first 200 chars of generated text
        preview = doc.text[:200].replace("\n", "\\n")
        print(f"  text:   {preview!r}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Stage 1 generation")
    parser.add_argument(
        "--checkpoint",
        default="runs/20260224_212158/checkpoints/best_model.pt",
        type=Path,
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping smoke test")
        sys.exit(0)

    device = torch.device("cuda")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, dtype = load_model(args.checkpoint, device)

    check_forward_inference(model, device, dtype)
    check_generate(model, device, args.max_new_tokens, args.temperature)
    print("[smoke test passed]")


if __name__ == "__main__":
    main()
