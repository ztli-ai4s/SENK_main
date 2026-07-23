from .registry import model_entrypoint
 
from .graph_attention_transformer import *
from .equiformer_v2_backbone import *  # V2 factories: equiformer_v2_l4_m2 etc.
# Keep a minimal set to avoid importing unmigrated modules.