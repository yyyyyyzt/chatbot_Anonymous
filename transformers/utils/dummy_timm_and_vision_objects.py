# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..file_utils import requires_backends


DETR_PRETRAINED_MODEL_ARCHIVE_LIST = None


class DetrForObjectDetection:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["timm", "vision"])

    def forward(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])


class DetrForSegmentation:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["timm", "vision"])

    def forward(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])


class DetrModel:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["timm", "vision"])

    def forward(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])


class DetrPreTrainedModel:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["timm", "vision"])

    def forward(self, *args, **kwargs):
        requires_backends(self, ["timm", "vision"])