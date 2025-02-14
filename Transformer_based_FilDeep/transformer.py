import torch.nn as nn
from core.cle_and_cld import CharacteristicLineEncoder, CharacteristicLineDecoder
from core.cse_and_csd import CrossSectionEncoder, CrossSectionDecoder
from core.fusion import WorkpieceRepresentationFusionEncoder, ExternalConditionFusionEncoder
from core.mpe import MotionParametersEncoder
from core.attention import CrossModule, SelfModule

class Transformer(nn.Module):
    def __init__(
            self,
            input_dim = 3,
            input_length = 300,
            seq_length = 60,
            emb_dim = 64,
            output_length = 300,
            cross_layers = 3,
            cross_heads = 4,
            cross_fea_len = 64,
            self_layers = 3,
            self_heads = 4,
            self_fea_len = 64,
            motion_degrees = 6
    ):
        super().__init__()
        self.cle_for_mould = CharacteristicLineEncoder(
            dim = input_dim,
            input_length = input_length,
            seq_length = seq_length,
            emb_dim = emb_dim
        )
        self.cle_for_strip = CharacteristicLineEncoder(
            dim = input_dim,
            input_length = input_length,
            seq_length = seq_length,
            emb_dim = emb_dim
        )
        self.cse = CrossSectionEncoder()
        self.csd = CrossSectionDecoder()
        self.wrfe = WorkpieceRepresentationFusionEncoder(cl_dim = emb_dim)
        self.mpe = MotionParametersEncoder(
            seq_length = seq_length,
            emb_dim = emb_dim,
            motion_degrees = motion_degrees
        )

        self.ecfe = ExternalConditionFusionEncoder()

        self.cross_attention = CrossModule(
            num_layers = cross_layers,
            num_heads = cross_heads,
            fea_len = cross_fea_len
        )
        self.self_attention = SelfModule(
            num_layers = self_layers,
            num_heads = self_heads,
            fea_len = self_fea_len
        )

        self.cld = CharacteristicLineDecoder(
            dim = input_dim,
            seq_length = seq_length,
            emb_dim = emb_dim,
            output_length = output_length
        )
        self._initialize()

    def forward(self, strip, mould, section, params):
        strip = self.cle_for_strip(strip)
        mould = self.cle_for_mould(mould)
        section = self.cse(section)
        params = self.mpe(params)
        recover_section = self.csd(section)
        strip = self.wrfe(strip, section)

        external_condition = self.ecfe(params, mould)
        strip = self.cross_attention(external_condition, strip)
        strip = self.self_attention(strip)
        strip = self.cld(strip)

        return recover_section, strip

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
