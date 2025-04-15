"""
Manifest generator for Speech-MASSIVE + MASSIVE datasets (per language and split).

This script reads MASSIVE annotations and Speech-MASSIVE audio metadata,
merges them per split, and outputs SpeechBrain-compatible manifest CSVs.

Expected columns in final manifest:
    ID, wav, utt, annot_utt, intent, split

Author: Haroun Elleuch, 2025
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def _load_massive_annotations(massive_jsonl_path: str) -> pd.DataFrame:
    """
    Load MASSIVE annotations from a JSONL file.

    Args:
        massive_jsonl_path: Path to the MASSIVE JSONL file.

    Returns:
        DataFrame with columns: id, utt, annot_utt, intent.
    """
    with open(massive_jsonl_path, "r", encoding="utf-8") as f:
        massive_data = [json.loads(line) for line in f]

    df = pd.DataFrame(massive_data)
    df = df[["id", "utt", "annot_utt", "intent"]]
    df["id"] = df["id"].astype(str)
    return df


def _load_and_merge_split(
        csv_path: str,
        massive_df: pd.DataFrame,
        split_name: str,
        wav_base_path: str,
) -> Optional[pd.DataFrame]:
    """
    Load a Speech-MASSIVE CSV and merge with MASSIVE annotations.

    Args:
        csv_path: Path to the Speech-MASSIVE CSV.
        massive_df: DataFrame with MASSIVE annotations.
        split_name: Name of the data split (e.g., train, dev).
        wav_base_path: Base path to the audio files.

    Returns:
        Merged DataFrame or None if file not found.
    """
    if not Path(csv_path).exists():
        logger.error(f"Missing file: {csv_path}")
        return None

    spm_df = pd.read_csv(csv_path)
    spm_df = spm_df[spm_df["is_validated"] == True]

    spm_df["id"] = spm_df["massive_id"].astype(str)
    spm_df["wav"] = spm_df["file_name"].apply(lambda x: str(Path(wav_base_path) / x))
    spm_df["split"] = split_name

    merged_df = spm_df.merge(massive_df, on="id", how="left")
    merged_df["ID"] = merged_df["id"]

    final_df = merged_df[["ID", "wav", "utt", "annot_utt", "intent", "split"]]
    return final_df


def build_per_split_manifests(
        lang: str,
        massive_jsonl_path: str,
        spm_csv_paths: Dict[str, str],
        wav_base_path: str,
        output_dir: str = "."
) -> None:
    """
    Create a separate SpeechBrain-compatible manifest CSV for each data split.

    Args:
        lang: Language code (e.g., 'fr-FR').
        massive_jsonl_path: Path to MASSIVE JSONL annotation file.
        spm_csv_paths: Dict mapping split names to CSV paths.
        wav_base_path: Directory containing WAV files.
        output_dir: Directory to save the generated manifest CSVs.
    """
    logger.info(f"Loading MASSIVE annotations for language: {lang}")
    massive_df = _load_massive_annotations(massive_jsonl_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split_name, csv_path in spm_csv_paths.items():
        logger.info(f"Processing split: {split_name}")
        split_df = _load_and_merge_split(csv_path, massive_df, split_name, wav_base_path)
        if split_df is not None:
            out_path = Path(output_dir) / f"manifest_{lang}_{split_name}.csv"
            split_df.to_csv(out_path, index=False)
            logger.info(f"✅ Saved: {out_path}")
        else:
            logger.warning(f"⚠️ Skipped: {split_name} (missing or empty)")


def prepare_dataset(
        lang: str,
        massive_data_root: str,
        speech_massive_root: str,
        output_dir: str = "manifests"
) -> None:
    """
    Prepare SpeechBrain-compatible manifests from MASSIVE and Speech-MASSIVE datasets.

    Args:
        lang: Language code (e.g., 'fr-FR').
        massive_data_root: Path to MASSIVE dataset root (should contain <lang>.jsonl).
        speech_massive_root: Path to Speech-MASSIVE dataset root (should contain train_dev/ and test/ folders).
        output_dir: Directory to save the resulting manifest CSVs.
    """
    massive_jsonl = Path(massive_data_root) / f"{lang}.jsonl"
    spm_train_dev = Path(speech_massive_root) / "train_dev" / lang
    spm_test = Path(speech_massive_root) / "Speech-MASSIVE-test" / "test" / lang
    wav_base_path = spm_train_dev / "audio"

    spm_csv_paths = {
        "train-115": spm_train_dev / "train-115.csv",
        "train": spm_train_dev / "train.csv",  # Optional
        "dev": spm_train_dev / "dev.csv",
        "test": spm_test / "test.csv",
    }

    build_per_split_manifests(
        lang=lang,
        massive_jsonl_path=str(massive_jsonl),
        spm_csv_paths={k: str(v) for k, v in spm_csv_paths.items()},
        wav_base_path=str(wav_base_path),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SpeechBrain-compatible manifest CSVs from MASSIVE and Speech-MASSIVE datasets."
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="fr-FR",
        help="Language code (e.g., 'fr-FR')"
    )

    parser.add_argument(
        "--massive_data_root",
        type=str,
        default="amazon-massive-dataset-1.0/data",
        help="Path to MASSIVE dataset root (should contain <lang>.jsonl)"
    )

    parser.add_argument(
        "--speech_massive_root",
        type=str,
        default="Speech-MASSIVE/Speech-MASSIVE",
        help="Path to Speech-MASSIVE dataset root (should contain train_dev/ and test/ folders)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="manifests",
        help="Directory where manifest CSVs will be saved (default: 'manifests')"
    )

    args = parser.parse_args()

    prepare_dataset(
        lang=args.lang,
        massive_data_root=args.massive_data_root,
        speech_massive_root=args.speech_massive_root,
        output_dir=args.output_dir
    )
