import argparse
import csv
import os
from logging import getLogger
from typing import List

from tqdm import tqdm

logger = getLogger(__name__)


def generate_manifest(
        data_root: str,
        output_dir: str,
        lang: str,
        splits: List[str] = ["train", "dev", "test"]
) -> None:
    """
    Generate SpeechBrain-compatible CSV manifests from FLEURS TSV files.

    Args:
        data_root (str): Path to the FLEURS dataset root directory.
        output_dir (str): Path where manifest CSVs will be saved.
        lang (str): Language folder name (e.g., 'ar_eg').
        splits (List[str], optional): List of splits to process. Defaults to ['train', 'dev', 'test'].
    """
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        input_tsv = os.path.join(data_root, lang, f"{split}.tsv")
        if not os.path.exists(input_tsv):
            logger.warning(f"Split not found: {input_tsv}")
            continue

        output_csv = os.path.join(output_dir, f"{lang}_{split}.csv")
        with open(input_tsv, 'r', encoding='utf-8') as fin, open(output_csv, 'w', encoding='utf-8', newline='') as fout:
            reader = csv.reader(fin, delimiter='\t')
            writer = csv.DictWriter(fout, fieldnames=["ID", "wav", "duration", "spk_gender", "transcript"])
            writer.writeheader()

            for row in tqdm(reader, desc=f"Processing {split}"):
                if len(row) < 7:
                    logger.warning(f"Skipping malformed row: {row}")
                    continue

                id_, filename, _, normalized_text, _, duration_frames, gender = row
                wav_path = os.path.abspath(os.path.join(data_root, lang, "wavs", filename.strip()))

                writer.writerow({
                    "ID": id_.strip(),
                    "wav": wav_path,
                    "duration": duration_frames.strip(), 
                    "spk_gender": gender.strip().upper(),
                    "transcript": normalized_text.strip()
                })

        logger.info(f"Manifest written: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SpeechBrain manifests from FLEURS dataset.")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DSDIR", "") + "/FLEURS",
                        help="Root directory of the FLEURS dataset (default: $DSDIR/FLEURS)")
    parser.add_argument("--output_dir", type=str, default="manifests",
                        help="Directory where manifest CSVs will be saved")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language code, e.g., 'ar_eg'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting manifest generation for language: {args.lang}")
    generate_manifest(args.data_root, args.output_dir, args.lang)
