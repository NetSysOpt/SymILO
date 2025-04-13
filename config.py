from utils import *
import os

confInfo = {
'IP': {
    'name':'IP',
    'trainDir':'./data/datasets/IP/train',
    'testDir':'./data/datasets/IP/test',
    'nGroup':16+16,
    'reorder':reorderIP,
    'addPosFeature':addPosFeatureIP
},

'SMSP':{
    'name':'SMSP',
    'trainDir':'./data/datasets/SMSP/train',
    'testDir':'./data/datasets/SMSP/train',
    'nGroup':16+16,
    'reorder':reorderSMSP,
    'addPosFeature':addPosFeatureSMSP
},


}

