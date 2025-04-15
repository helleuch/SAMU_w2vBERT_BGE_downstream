import argparse
import os
from logging import getLogger, basicConfig, INFO
from typing import List

import pandas as pd
from tqdm import tqdm

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def generate_combined_manifests(
        data_root: str,
        output_dir: str,
        splits: List[str] = ["train", "dev", "test"]
) -> None:
    """
    Generate combined SpeechBrain-compatible CSV manifests from FLEURS dataset.

    Args:
        data_root (str): Path to the FLEURS dataset root directory.
        output_dir (str): Path where manifest CSVs will be saved.
        splits (List[str], optional): List of splits to process. Defaults to ['train', 'dev', 'test'].
    """
    os.makedirs(output_dir, exist_ok=True)

    langs = [d for d in os.listdir(data_root)
             if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".IDRIS")]
    logger.info(f"Found {len(langs)} languages: {langs}")

    for split in splits:
        combined_rows = []

        for lang in tqdm(langs, desc=f"Processing split: {split}"):
            input_tsv = os.path.join(data_root, lang, f"{split}.tsv")
            if not os.path.exists(input_tsv):
                logger.warning(f"Missing {split} for {lang}: {input_tsv}")
                continue

            try:
                df = pd.read_csv(input_tsv, sep='\t', header=None, quoting=3, dtype=str)
            except Exception as e:
                logger.warning(f"Error reading {input_tsv}: {e}")
                continue

            if df.shape[1] < 7:
                logger.warning(f"Malformed data in {input_tsv}, expected at least 7 columns")
                continue

            df.columns = ["ID", "filename", "unused1", "normalized_text", "unused2", "duration", "gender"]
            df["wav"] = df["filename"].apply(
                lambda x: os.path.abspath(os.path.join(data_root, lang, "audio", split, x.strip())))
            df["spk_gender"] = df["gender"].str.upper()
            df["transcript"] = df["normalized_text"].str.strip()

            df["ID"] = lang + "_" + df["ID"] + "_" + df["filename"].apply(lambda x: x.replace(".wav",""))

            df = df[["ID", "wav", "duration", "spk_gender", "transcript"]]
            combined_rows.append(df)

        if combined_rows:
            combined_df = pd.concat(combined_rows, ignore_index=True)
            output_csv = os.path.join(output_dir, f"{split}.csv")
            combined_df.to_csv(output_csv, index=False)
            logger.info(f"Wrote {len(combined_df)} entries to {output_csv}")

            if combined_df.ID.nunique() != len(combined_df):
                logger.warning(
                    f"Duplicate IDs found in the {split} manifest after language prefixing: number of unique IDs: {combined_df.ID.nunique()}," \
                    f"total numbeer of rows: {len(combined_df)} " \
                    f"total number of unique audios: {combined_df.wav.nunique()}")

        else:
            logger.warning(f"No data collected for split: {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined SpeechBrain manifests from FLEURS dataset.")
    parser.add_argument("--data_root", type=str, default=os.environ.get("DSDIR", "") + "/FLEURS",
                        help="Root directory of the FLEURS dataset (default: $DSDIR/FLEURS)")
    parser.add_argument("--output_dir", type=str, default="manifests",
                        help="Directory where manifest CSVs will be saved")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting manifest generation for all languages.")
    generate_combined_manifests(args.data_root, args.output_dir)
