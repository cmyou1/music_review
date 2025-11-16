"""
Generate a simple sine-wave WAV file for testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def generate(output: Path, duration: float = 5.0, freq: float = 440.0, sr: int = 32000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.25 * np.sin(2 * np.pi * freq * t)
    sf.write(output, waveform, sr)
    print(f"샘플 오디오 생성 완료: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="간단한 사인파 WAV 파일 생성")
    parser.add_argument("--output", type=Path, default=Path("data/sample_songs/tone.wav"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--freq", type=float, default=440.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.output, duration=args.duration, freq=args.freq)

