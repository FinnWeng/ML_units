import argparse
import os
import sys


from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C

# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()

# Model type
# _C.MODEL.TYPE = ""
_C.MODEL.TYPE = "regnet"

# # Number of weight layers
# _C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 128

# # Loss function (see pycls/models/loss.py for options)
# _C.MODEL.LOSS_FUN = "cross_entropy"

# ------------------------------------------------------------------------------------ #
# RegNet options
# ------------------------------------------------------------------------------------ #
_C.REGNET = CfgNode()

# Stem type
_C.REGNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.REGNET.STEM_W = 32

# Block type
_C.REGNET.BLOCK_TYPE = "res_bottleneck_block"

# Stride of each stage
_C.REGNET.STRIDE = 2

# Squeeze-and-Excitation (RegNetY)
_C.REGNET.SE_ON = True
_C.REGNET.SE_R = 0.25

# Depth
_C.REGNET.DEPTH = 6

# Initial width
_C.REGNET.W0 = 16

# Slope
_C.REGNET.WA = 36.44

# Quantization
_C.REGNET.WM =  2.49

# Group width
_C.REGNET.GROUP_W = 8

# Bottleneck multiplier (bm = 1 / b from the paper)
_C.REGNET.BOT_MUL = 1.0



# ------------------------------------------------------------------------------------ #
# AnyNet options
# ------------------------------------------------------------------------------------ #
_C.ANYNET = CfgNode()

# Stem type
_C.ANYNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.ANYNET.STEM_W = 32

# Block type
_C.ANYNET.BLOCK_TYPE = "res_bottleneck_block"

# Depth for each stage (number of blocks in the stage)
_C.ANYNET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.ANYNET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.ANYNET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.ANYNET.BOT_MULS = []

# Group widths for each stage (applies to bottleneck block)
_C.ANYNET.GROUP_WS = []

# Whether SE is enabled for res_bottleneck_block
_C.ANYNET.SE_ON = False

# SE ratio
_C.ANYNET.SE_R = 0.25






# ------------------------------------------------------------------------------------ #
# Batch norm options
# ------------------------------------------------------------------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0