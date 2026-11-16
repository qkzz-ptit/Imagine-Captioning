"""Fine-tune an existing caption model on one image/caption pair.

This creates a separate checkpoint so a one-image personalization does not
overwrite the general-purpose model.
"""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from nltk.tokenize import word_tokenize

from src.beam_search import beam_search
from src.model import CaptionModel
from src.transform import get_eval_transform
from src.utils import load_checkpoint


def tokenize_caption(caption):
    """Use NLTK's bundled tokenizer without requiring downloaded punkt data."""
    return word_tokenize(caption.lower(), preserve_line=True)


def extend_vocabulary(vocab, caption):
    """Append missing lower-case tokens without changing any existing IDs."""
    for token in tokenize_caption(caption):
        if token not in vocab.word2idx:
            index = len(vocab)
            vocab.word2idx[token] = index
            vocab.idx2word[index] = token


def build_model(checkpoint, device):
    """Restore matching weights without downloading ViT weights."""
    model_config = dict(checkpoint["model_config"])
    model_config["vocab_size"] = len(checkpoint["vocab"])

    import torchvision.models as torchvision_models

    vit_b_16 = torchvision_models.vit_b_16

    def without_pretrained_weights(*args, **kwargs):
        kwargs["weights"] = None
        kwargs.pop("pretrained", None)
        return vit_b_16(*args, **kwargs)

    with patch.object(
        torchvision_models, "vit_b_16", new=without_pretrained_weights
    ):
        model = CaptionModel(**model_config)

    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()
    for name, tensor in old_state.items():
        if name in ("decoder.embed.weight", "decoder.fc.weight"):
            continue
        if name in new_state and new_state[name].shape == tensor.shape:
            new_state[name].copy_(tensor)

    old_embedding = old_state["decoder.embed.weight"]
    with torch.no_grad():
        model.decoder.embed.weight[: old_embedding.size(0)].copy_(old_embedding)
        nn.init.normal_(
            model.decoder.embed.weight[old_embedding.size(0) :], mean=0.0, std=0.02
        )

    model.load_state_dict(new_state)
    return model.to(device), model_config


def caption_ids(model, features, vocab, max_len):
    model.eval()
    with torch.inference_mode():
        ids = model.decoder.sample(features, max_len=max_len)[0]
    return [token for token in ids if token != vocab.word2idx["<end>"]]


def main():
    parser = argparse.ArgumentParser(
        description="Personalize an image caption model for one image and sentence."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/personalized_caption.pth")
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    base_path = args.base_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}; pass --overwrite to replace it."
        )
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    checkpoint = load_checkpoint(base_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    extend_vocabulary(vocab, args.caption)
    tokens = tokenize_caption(args.caption)
    target_ids = [vocab.word2idx["<start>"]] + [vocab.word2idx[t] for t in tokens] + [
        vocab.word2idx["<end>"]
    ]
    target = torch.tensor([target_ids], dtype=torch.long, device=device)

    model, model_config = build_model(checkpoint, device)
    model.eval()
    transform = get_eval_transform(args.image_size)
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(image_tensor)
    features = features.detach()

    # The vision encoder stays fixed; only the caption decoder is personalized.
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    generated = []
    for step in range(1, args.max_steps + 1):
        model.decoder.train()
        logits, _ = model.decoder(features, target)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            generated = caption_ids(model, features, vocab, max_len=len(tokens) + 3)
            rendered = " ".join(vocab.idx2word[index] for index in generated)
            print(
                f"step={step}/{args.max_steps} loss={loss.item():.4f} "
                f"greedy={rendered}",
                flush=True,
            )
            if generated == target_ids[1:-1]:
                print(f"Greedy decoding matched the target at step {step}.", flush=True)
                break

    greedy_ids = caption_ids(model, features, vocab, max_len=len(tokens) + 3)
    model.eval()
    with torch.inference_mode():
        beam_ids = beam_search(
            model.decoder,
            features,
            vocab,
            beam_size=args.beam_size,
            max_len=len(tokens) + 3,
            return_alphas=False,
        )
    if isinstance(beam_ids, tuple):
        beam_ids = beam_ids[0]
    beam_ids = [int(index) for index in beam_ids if int(index) != vocab.word2idx["<end>"]]

    greedy_text = " ".join(vocab.idx2word[index] for index in greedy_ids)
    beam_text = " ".join(vocab.idx2word[index] for index in beam_ids)
    print(f"Final greedy: {greedy_text}", flush=True)
    print(f"Final beam:   {beam_text}", flush=True)

    model_config["vocab_size"] = len(vocab)
    personalized_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "vocab": vocab,
        "caption": args.caption,
        "training_image": str(image_path),
        "base_checkpoint": str(base_path),
        "personalization": "single_image_caption_finetune",
        "display_word_overrides": {
            token: token[0].upper() + token[1:]
            for token in tokens
            if token in {"mr.", "pham", "minh", "tuyen"}
        },
        "fine_tuning_steps": step,
        "inference_greedy_caption": greedy_text,
        "inference_beam_caption": beam_text,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb", dir=output_path.parent, suffix=".tmp", delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        torch.save(personalized_checkpoint, temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved personalized checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
