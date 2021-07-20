from modeling_utils_neo import GediMixin
from transformers import GPTNeoModel

class GediGPTNeoModel(GPTNeoModel, GediMixin):
    gedi_mode = True
