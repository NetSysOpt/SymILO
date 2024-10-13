from utils import *
import os

confInfo = {
'IP': {
    'name':'IP',
    'trainDir':r'../data_gen/IP/train',
    'testDir':r'../data_gen/IP/test',
    'nGroup':16+16,
    'reorder':reorderIP,
    'addPosFeature':addPosFeatureIP
},

'IP_M': {
    'name':'IP_M',
    'testDir':r'../data_gen/IP/10_150_5',
    'nGroup':16+16,
    'reorder':reorderIP,
    'addPosFeature':addPosFeatureIP
},

'IP_L': {
    'name':'IP_L',
    'testDir':r'../data_gen/IP/15_150_8',
    'nGroup':16+16,
    'reorder':reorderIP,
    'addPosFeature':addPosFeatureIP
},

'SMSP':{
    'name':'SMSP',
    'trainDir':r'F:\L2O_project\Neurips2023\exps\data\smsp\train',
    'testDir':r'F:\L2O_project\Neurips2023\exps\data\smsp\tune',
    'nGroup':16+16,
    'reorder':reorderSMSP,
    'addPosFeature':addPosFeatureSMSP
},


}


#
# info = ipTuneInfo
#
# DIR_INS = os.path.join(info['trainDir'],'ins')
# DIR_SOL = os.path.join(info['trainDir'],'sol')
# DIR_BG = os.path.join(info['trainDir'],'bg')
# NGROUP = info['nGroup']
#
# TEST_INS = os.path.join(info['testDir'],'ins')
# TEST_BG = os.path.join(info['testDir'],'bg')
#
# ADDPOS = info['addPosFeature']
# REORDER = info['reorder']
#
#
# DATA_NAME = info['name']