"""python -m mmada_space.understanding image.jpg "Describe this image." --layers 0 15 31"""
import argparse
from pathlib import Path

from .adapter import MMaDAAdapter
from .capture import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('question')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 15, 31])
    parser.add_argument('--capture-step', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs/understanding'))
    args = parser.parse_args()
    adapter = MMaDAAdapter()
    input_ids = adapter.prepare_understanding(args.image, args.question)
    with Capture(adapter, args.layers, step=args.capture_step) as captured:
        answer = adapter.answer(
            input_ids, max_new_tokens=args.max_new_tokens, steps=args.steps,
            block_length=args.block_length, seed=args.seed,
        )
    metadata = dict(
        **adapter.metadata(), task='understanding', image=str(args.image.resolve()),
        question=args.question, answer=answer, seed=args.seed, steps=args.steps,
        max_new_tokens=args.max_new_tokens, block_length=args.block_length,
        temperature=0.0, cfg_scale=0.0, remasking='low_confidence',
        alignment=adapter.alignment(captured.input_ids, input_ids.shape[1]),
        preprocessing='RGB; bicubic shortest-side resize; 512 center crop; [-1, 1]',
    )
    captured.save(args.output, metadata)
    (Path(args.output) / 'answer.txt').write_text(answer + '\n', encoding='utf-8')
    print(f'U | capture step {args.capture_step}/{captured.evaluations} | layers {list(args.layers)}')
    print(answer)


if __name__ == '__main__':
    main()
