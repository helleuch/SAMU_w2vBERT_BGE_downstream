"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Salima Mdhaffar 2024
"""

import torch
from torch import nn

try:
    from transformers import (HubertConfig, HubertModel, Wav2Vec2BertConfig,
                              Wav2Vec2BertModel, Wav2Vec2Config,
                              Wav2Vec2FeatureExtractor, Wav2Vec2Model)

except ImportError:
    print(
        "Please install transformer from HuggingFace to use wav2vec2/Hubert !"
    )

HF_models = {"wav2vec2": Wav2Vec2Model, "hubert": HubertModel, "w2v_bert": Wav2Vec2BertModel}
HF_config = {"wav2vec2": Wav2Vec2Config, "hubert": HubertConfig, "w2v_bert": Wav2Vec2BertConfig}


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=False,
        freeze_feature_extractor=False,
        pretrain=True,
        output_all_hiddens=False,
        apply_spec_augment=False,
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        elif "w2v-bert" in source:
            config = HF_config.get("w2v_bert")
            model = HF_models.get("w2v_bert")
        else:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")

        # Download the model from HuggingFace.
        # if pretrain is False, we do not download the pretrained weights
        # it it is True, we download and load them.
        if not (pretrain):
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
        else:
            self.model = model.from_pretrained(source, cache_dir=save_path)

        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()

        # processor = AutoProcessor.from_pretrained(source)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        outputs = self.model(wav, output_hidden_states=True)
        if self.output_all_hiddens:
            outputs = torch.stack(outputs['hidden_states'][:-1], dim=0)
        else:
            outputs = outputs['last_hidden_state']
        return outputs
