import torch
import torch.nn as nn

from core.attention import AddNorm, Attention_Cross_Fidelity
from Transformer_based_FilDeep.transformer import Transformer

class BasicBlock(torch.nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 activation: type[torch.nn.Module]):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features)
        self.activation = activation()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.activation(x)
        return x

class FilDeep(nn.Module):
    def __init__(
            self,
            is_frozen_CL=True,
            is_frozen_CS=True,
            is_frozen_MP=True,
            is_frozen_WRFE=True,
            is_frozen_ECFE=True,
            is_frozen_low_Attention=True,
            input_dim=3,
            input_length=300,
            seq_length=60,
            emb_dim=64,
            output_length=300,
            cross_layers = 3,
            cross_heads = 4,
            cross_fea_len = 64,
            self_layers = 3,
            self_heads = 4,
            self_fea_len = 64,
            motion_degrees = 6,
            acf_layers=2,
            acf_heads=4,
            acf_fea_len=64,
    ):
        super().__init__()
        self.is_freeze_CL = is_frozen_CL
        self.is_freeze_CS = is_frozen_CS
        self.is_freeze_MP = is_frozen_MP
        self.is_freeze_WRFE = is_frozen_WRFE
        self.is_freeze_ECFE = is_frozen_ECFE
        self.is_freeze_low_DP = is_frozen_low_Attention


        self.low = Transformer(
            input_dim=input_dim,
            input_length=input_length,
            seq_length=seq_length,
            emb_dim=emb_dim,
            output_length=output_length,
            cross_layers=cross_layers,
            cross_heads=cross_heads,
            cross_fea_len=cross_fea_len,
            self_layers=self_layers,
            self_heads=self_heads,
            self_fea_len=self_fea_len,
            motion_degrees=motion_degrees
        )


        self.acf = Attention_Cross_Fidelity(
            num_layers=acf_layers,
            num_heads=acf_heads,
            fea_len=acf_fea_len
        )

        self.add_norm = AddNorm(emb_dim, 0)

        self._initialize()

    def _freeze(self, _model):
        for param in _model.parameters():
            param.requires_grad = False

    def _thaw(self, _model):
        for param in _model.parameters():
            param.requires_grad = True

    def freeze_low(self, layer, is_frozen_CL, is_frozen_CS, is_frozen_MP, is_frozen_WRFE,is_freeze_ECFE , is_frozen_low_Attention):
        if is_frozen_CL:
            self._freeze(layer.cle_for_mould)
            self._freeze(layer.cle_for_strip)
            self._freeze(layer.cld)

        if is_frozen_CS:
            self._freeze(layer.cse)
            self._freeze(layer.csd)

        if is_frozen_MP:
            self._freeze(layer.mpe)

        if is_frozen_WRFE:
            self._freeze(layer.wrfe)

        if is_freeze_ECFE:
            self._freeze(layer.ecfe)

        if is_frozen_low_Attention:
            self._freeze(layer.cross_attention)
            self._freeze(layer.self_attention)

    def thaw_low(self, layer, is_frozen_CL, is_frozen_CS, is_frozen_MP, is_frozen_WRFE, is_freeze_ECFE, is_frozen_low_Attention):
        if is_frozen_CL:
            self._thaw(layer.cle_for_mould)
            self._thaw(layer.cle_for_strip)
            self._thaw(layer.cld)

        if is_frozen_CS:
            self._thaw(layer.cse)
            self._freeze(layer.cse.section_encoder)
            self._thaw(layer.csd)

        if is_frozen_MP:
            self._thaw(layer.mpe)

        if is_frozen_WRFE:
            self._thaw(layer.wrfe)

        if is_freeze_ECFE:
            self._thaw(layer.ecfe)

        if is_frozen_low_Attention:
            self._thaw(layer.cross_attention)
            self._thaw(layer.self_attention)

    # old
    def forward(self, strip, mould, section, params, fidelity):
        if not fidelity:
            self.thaw_low(self.low, self.is_freeze_CL, self.is_freeze_CS, self.is_freeze_MP, self.is_freeze_WRFE, self.is_freeze_ECFE, self.is_freeze_low_DP)
            return self.low(strip, mould, section, params)
        else:
            self.freeze_low(self.low, self.is_freeze_CL, self.is_freeze_CS, self.is_freeze_MP, self.is_freeze_WRFE, self.is_freeze_ECFE, self.is_freeze_low_DP)
            strip = self.low.cle_for_strip(strip)
            mould = self.low.cle_for_mould(mould)
            section = self.low.cse(section)
            strip = self.low.wrfe(strip, section)
            params = self.low.mpe(params)

            external_condition = self.low.ecfe(params, mould)

            low_loaded_strip_feature = self.low.cross_attention(external_condition, strip)
            low_loaded_strip_feature = self.low.self_attention(low_loaded_strip_feature)


            strip = self.acf(external_condition, low_loaded_strip_feature)
            strip = self.add_norm(low_loaded_strip_feature, strip)
            strip = self.low.cld(strip)

            return strip

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain("relu"))
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0)
        return True

