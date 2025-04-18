import sys
from logging import INFO

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(INFO)

# Define training procedure.


class SLU(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from waveform to output probabilities."""

        # Get data.
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Forward pass.
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # Output layer for seq2seq log-probabilities.
        logits = self.modules.output_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        # Get data.
        batch = batch.to(self.device)
        chars, char_lens = batch.char_encoded
        ids = batch.id

        # Get predictions & loss.
        p_ctc, wav_lens = predictions
        loss = self.hparams.ctc_cost(p_ctc, chars, wav_lens, char_lens)

        # Get metrics.
        if stage != sb.Stage.TRAIN:
            # Generate sequences with CTC greedy decoder.
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, self.hparams.blank_index
            )
            # Update metrics.
            self.cer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.tokenizer.decode_ndim,
            )
            self.coer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.tokenizer.decode_ndim,
            )
            self.cver_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.tokenizer.decode_ndim,
            )
            self.ctc_metric.append(ids, p_ctc, chars, wav_lens, char_lens)

        return loss

    def init_optimizers(self):
        """Initializes the wav2vec2 optimizer and model optimizer"""

        # Join optimizers.
        self.optimizer_wav2vec = self.hparams.opt_class_wav2vec(
            self.hparams.model_wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        # Add optimizers to checkpoint recoverables.
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "optimizer_wav2vec", self.optimizer_wav2vec
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "wav2vec_optimizer": self.optimizer_wav2vec,
            "model_optimizer": self.optimizer,
        }

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

        # Re-initialize metrics.
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.ctc_metric = self.hparams.ctc_computer()
            self.coer_metric = self.hparams.coer_computer()
            self.cver_metric = self.hparams.cver_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""

        # Save loss and metrics.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["COER"] = self.coer_metric.summarize("error_rate")
            stage_stats["CVER"] = self.cver_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer_wav2vec, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"]},
                min_keys=["CER"],
            )

        # Same plus write results in txt files.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(hparams["cer_file_test"], "w", encoding="utf-8") as w:
                self.cer_metric.write_stats(w)
            with open(hparams["ctc_file_test"], "w", encoding="utf-8") as w:
                self.ctc_metric.write_stats(w)
            with open(hparams["coer_file_test"], "w", encoding="utf-8") as w:
                self.coer_metric.write_stats(w)
            with open(hparams["cver_file_test"], "w", encoding="utf-8") as w:
                self.cver_metric.write_stats(w)


# Define custom data procedure.
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets:
    csv_folder = hparams["save_folder"] + "/csv"

    if hparams.get("few_shot", False):
        logger.info(f"Loading few shot CSV...")
        train_csv = hparams["csv_train_115"]
    else:
        logger.info(f"Loading complete train CSV...")
        train_csv = hparams["csv_train"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv,
        replacements={"data_root": csv_folder},
    )

    # We sort training data to speed up training and get better results.
    # When sorting do not shuffle in dataloader ! otherwise is pointless.
    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # We also sort the validation data so it is faster to validate.
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": csv_folder}
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration", reverse=True)

    # We also sort the test data so it is faster to validate.
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": csv_folder}
    )
    test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)

    datasets = [train_data, valid_data, test_data]

    print("Creating a CTC Text Encoder object...")
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides("char_list", "char_encoded")
    def text_pipeline(char):
        char_list = char.strip().split()
        yield char_list
        char_encoded = label_encoder.encode_sequence_torch(char_list)
        yield char_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Create a label encoder instead of a tokenizer for our tag list:
    print("Trying to save Label Encoder...")
    lab_enc_file = hparams["save_folder"] + "/labelencoder.txt"
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    # 5. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "char_encoded"],
    )

    # 6. Make DataLoaders and shuffle if needed:
    dataloader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams["batch_size"],
        num_workers=hparams["dataloader_options"]["num_workers"],
        collate_fn=PaddedBatch,
        shuffle=hparams["dataloader_options"]["shuffle"],
    )
    dataloader_valid = torch.utils.data.DataLoader(
        valid_data,
        batch_size=hparams["batch_size"],
        num_workers=hparams["dataloader_options"]["num_workers"],
        collate_fn=PaddedBatch,
        shuffle=hparams["dataloader_options"]["shuffle"],
    )
    dataloader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=hparams["test_batch_size"],
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        collate_fn=PaddedBatch,
    )

    return dataloader_train, dataloader_valid, dataloader_test, label_encoder


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol.
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory.
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation:
    if hparams["do_preparation"]:
        from prepare_monolingual_speech_massive import prepare_dataset
        run_on_main(
            prepare_dataset,
            kwargs=hparams["data_prep_args"]
        )

    # Create the datasets objects as well as tokenization and encoding.
    train_data, valid_data, test_data, label_encoder = dataio_prepare(hparams)

    # Trainer initialization.
    slu_brain = SLU(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    slu_brain.tokenizer = label_encoder
    slu_brain.tokenizer.add_unk()  # handle unknown SLU labels

    # Load a pretrained SAMU-w2vBERT-BGE model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # Check for stopped training.
    slu_brain.checkpointer.recover_if_possible()

    # Train.
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        progressbar=True,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test.
    slu_brain.evaluate(
        test_data,
        min_key="CER",
        progressbar=True,
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
