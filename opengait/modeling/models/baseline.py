import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


class Baseline(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]
        """
        train:
        print (sils.size(),outs.size())
        torch.Size([128, 1, 30, 64, 44]) torch.Size([128, 256, 30, 16, 11])
        
        val:
        print(sils.size(), outs.size())   
        torch.Size([1, 1, 558, 64, 44]) torch.Size([1, 256, 558, 16, 11])
        torch.Size([1, 1, 609, 64, 44]) torch.Size([1, 256, 609, 16, 11])
        torch.Size([1, 1, 589, 64, 44]) torch.Size([1, 256, 589, 16, 11])
        torch.Size([1, 1, 730, 64, 44]) torch.Size([1, 256, 730, 16, 11])
        torch.Size([1, 1, 694, 64, 44]) torch.Size([1, 256, 694, 16, 11])
        torch.Size([1, 1, 797, 64, 44]) torch.Size([1, 256, 797, 16, 11])
        """
        
        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        """
        train:
        torch.Size([128, 256, 16, 11])
        
        val:
        print(outs.shape)
        torch.Size([16, 256, 16, 11])
        """

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]
        """
        print(feat.shape)
        torch.Size([16, 256, 31])
        """
        
        embed_1 = self.FCs(feat)  # [n, c, p]
        """
        print(embed_1.shape)
        torch.Size([16, 256, 31])
        """
        
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        """
        print(embed_2.shape, logits.shape)
        torch.Size([16, 256, 31]) torch.Size([16, 74, 31])
        """

        embed = embed_1

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
