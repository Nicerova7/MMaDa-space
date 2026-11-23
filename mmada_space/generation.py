"""python -m mmada_space.generation "A photograph of a dog." --layers 0 15 31"""
import argparse
from pathlib import Path

from .adapter import MMaDAAdapter
from .capture import Capture


def run(adapter, prompt, layers=(0, 15, 31), capture_step=1,
        steps=15, guidance=3.5, seed=42, output="outputs/generation"):
    with Capture(adapter, layers, step=capture_step) as captured:
        image = adapter.generate(prompt, steps=steps, guidance=guidance, seed=seed)
    metadata = dict(
        **adapter.metadata(), task="generation", prompt=prompt, steps=steps,
        guidance=guidance, seed=seed, temperature=1.0, mask_schedule="cosine",
        cfg_formula="(1 + guidance) * conditional - guidance * unconditional",
        alignment=adapter.alignment(captured.input_ids, len(captured.input_ids)),
    )
    captured.save(output, metadata)
    image.save(Path(output) / "image.png")
    print(f"G | capture step {capture_step}/{captured.evaluations} | layers {list(layers)}")
    print(f"Saved: {output}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/generation"))
    args = parser.parse_args()
    run(MMaDAAdapter(), **vars(args))


if __name__ == "__main__":
    main()
