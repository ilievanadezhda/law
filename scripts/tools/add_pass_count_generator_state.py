"""Add LAW's pass_count_generator_state to a Duo checkpoint so that LAW can load it.

Usage:
    python scripts/tools/add_pass_count_generator_state.py duo.ckpt duo-law.ckpt
"""
import argparse
import sys
from pathlib import Path

# The checkpoint pickles objects from the repo's modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    assert "pass_count_generator_state" not in ckpt
    # Same generator as LAW_BASE.__init__ (algo.py), seeded with config.seed.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    ckpt["pass_count_generator_state"] = generator.get_state()
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst} (global_step={ckpt['global_step']})")


if __name__ == "__main__":
    main()
