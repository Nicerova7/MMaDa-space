"""python -m mmada_space.understanding image.jpg "Describe this image." --layers 0 15 31"""
import argparse
from pathlib import Path

from .adapter import MMaDAAdapter
from .capture import Capture


def run(adapter, image, question, layers=(0, 15, 31), capture_step=1,
        max_new_tokens=128, steps=64, block_length=32, seed=42,
        output="outputs/understanding"):
    input_ids = adapter.prepare_understanding(image, question)
    with Capture(adapter, layers, step=capture_step) as captured:
        answer = adapter.answer(input_ids, max_new_tokens=max_new_tokens,
                                steps=steps, block_length=block_length, seed=seed)
    metadata = dict(
        **adapter.metadata(), task="understanding", image=str(Path(image).resolve()),
        question=question, answer=answer, seed=seed, steps=steps,
        max_new_tokens=max_new_tokens, block_length=block_length,
        temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
        alignment=adapter.alignment(captured.input_ids, input_ids.shape[1]),
        preprocessing="RGB; bicubic shortest-side resize; 512 center crop; [-1, 1]",
    )
    captured.save(output, metadata)
    (Path(output) / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    print(f"U | capture step {capture_step}/{captured.evaluations} | layers {list(layers)}")
    print(answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/understanding"))
    args = parser.parse_args()
    run(MMaDAAdapter(), **vars(args))


if __name__ == "__main__":
    main()
