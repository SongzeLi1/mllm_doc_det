import torch
import torch.nn as nn

from .resnet import resnet50


class NPRModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.module = resnet50(pretrained=pretrained, num_classes=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, image, label=None, *args, **kwargs):
        hidden, self.output = self.module(image)
        if label is None:
            return hidden

        long_label = label.type(torch.FloatTensor).to(self.output.device)
        loss = self.loss_fn(self.output.squeeze(1), long_label)
        output_label = self.output.sigmoid().flatten()

        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": output_label,
            # -----------------------------------------
        }
        output_dict = {
            # loss for backward
            "backward_loss": loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": None,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": output_label,
            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {"predict_loss": loss},
            "visual_image": {},
            # -----------------------------------------
        }
        return output_dict
