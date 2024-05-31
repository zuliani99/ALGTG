
from models.modules.GTG_Cls import GTGModule
from models.modules.LossNet import LossNet
#from models.backbones.ssd_pytorch.SSD import SSD, build_ssd
from models.backbones.ResNet18 import ResNet, ResNet18
from models.backbones.VGG import VGG, VGG16_bn

from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.baselines.LearningLoss import LearningLoss
from strategies.competitors.CoreSet import CoreSet
from strategies.competitors.BADGE import BADGE
from strategies.competitors.BALD import BALD
from strategies.competitors.CDAL import CDAL
from strategies.competitors.TA_VAAL.TA_VAAL import TA_VAAL
from strategies.competitors.AlphaMix import AlphaMix
from strategies.competitors.TiDAL import TiDAL
from strategies.GTG import GTG
from strategies.GTG_off import GTG_off

from models.BBone_Module import Master_Model

from datasets_creation.Classification import Cls_Datasets
from datasets_creation.Detection import Det_Dataset

#from config import voc_config
from typing import Dict, Any, List, Tuple


def get_dataset(task: str, dataset_name: str, init_lab_obs: int) -> Cls_Datasets | Det_Dataset:
    if task == 'clf': return Cls_Datasets(dataset_name, init_lab_obs=init_lab_obs)
    else: return Det_Dataset(dataset_name, init_lab_obs=init_lab_obs)

    
def get_backbone(n_classes: int, n_channels: int, bbone: str) -> ResNet | VGG:# | SSD:
    if bbone == 'ResNet': return ResNet18(n_classes=n_classes, n_channels=n_channels)
    else: return VGG16_bn(n_classes=n_classes,  n_channels=n_channels) #bbone == 'VGG':
    #else: return build_ssd('train', voc_config, num_classes=n_classes)
    

def get_module(module: str, module_params: Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]) -> LossNet | GTGModule:
    if module == 'LL': return LossNet(module_params)
    else: return GTGModule(module_params)


def get_ll_module_params(task: str, image_size: int, dataset_name: str) -> Dict[str, int | str | List[int]]:
    if task == 'detection':
        return dict(
                    feature_sizes=[512, 1024, 512, 256, 256, 256],
                    num_channels=[512, 1024, 512, 256, 256, 256],
                    interm_dim=128, task='detection'
                )
    else:
        if dataset_name != 'caltech256':   
            return dict(
                    feature_sizes=[image_size, image_size // 2, image_size // 4, image_size // 8],
                    num_channels=[64, 128, 256, 512],
                    interm_dim=128, task='cls'
                )
        else: 
            return dict(
                    feature_sizes=[224, 224, 112, 56],
                    num_channels=[64, 64, 128, 256],
                    interm_dim=128, task='cls'
                )
            
            

dict_strategies = dict(
    random = Random, entropy = Entropy, coreset = CoreSet, badge = BADGE, bald = BALD, # -> BB
    cdal = CDAL, gtg = GTG_off, alphamix = AlphaMix, # -> BB
    
    ll = LearningLoss, ll_gtg = GTG_off, tavaal = TA_VAAL, tidal = TiDAL,# -> BB + LL
    lq_gtg = GTG # -> BB + GTG
)

dict_backbone = dict(
    cifar10 = 'ResNet', cifar100 = 'ResNet', svhn = 'ResNet', fmnist = 'ResNet',
    caltech256 = 'VGG', tinyimagenet = 'ResNet', voc = 'SSD', coco = 'SSD'
)


def get_strategies_object(methods: List[str], Masters: Dict[str, Master_Model], 
                          ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> List[Any]:
    strategies: List[Random | Entropy | CoreSet | AlphaMix | BADGE | BALD | CDAL | GTG_off | LearningLoss | GTG_off | TiDAL | TA_VAAL | GTG] = []
    
    for method in methods:
        if 'gtg' in method.split('_'):
            
            for id_am_ts in range(len(gtg_p['am_ts'])):
                
                for id_am in range(len(gtg_p['am'])):
                    if method == 'gtg': m_model = Masters['M_None']
                    elif method.split('_')[0] == 'lq': m_model = Masters['M_GTG']
                    else: m_model = Masters['M_LL']

                    strategies.append(dict_strategies[method](
                        {**ct_p, 'Master_Model': m_model}, t_p, al_p, {**gtg_p, 'id_am_ts': id_am_ts, 'id_am': id_am})
                    )
                
        elif method in ['ll', 'tavaal', 'tidal']:
            strategies.append(dict_strategies[method]({ **ct_p, 'Master_Model': Masters['M_LL'] if method != 'tidal' else Masters['M_LL_tidal']}, t_p, al_p))
        else:
            strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': Masters['M_None']}, t_p, al_p))
    
    return strategies


# to create a single master model for each type
def get_masters(methods: List[str], BBone: ResNet | VGG,
                ll_module_params: Dict[str, Any], gtg_module_params: Dict[str, Any],
                dataset_name: str, n_classes: int) -> Dict[str, Master_Model]:
    
    ll, ll_tidal, only_bb = False, False, False
    
    Masters = { }
    
    # create and save the initial checkpoints of the masters
    for method in methods:
        if (method in ['ll', 'll_gtg', 'tavaal']) and not ll:
            Masters['M_LL'] = Master_Model(BBone, get_module('LL', {**ll_module_params, 'module_out': 1}), dataset_name)
            ll = True
        elif method == 'tidal' and not ll_tidal:
            Masters['M_LL_tidal'] = Master_Model(BBone, get_module('LL', {**ll_module_params, 'module_out': n_classes}), dataset_name)
            ll_tidal = True
        elif method == 'lq_gtg':
            Masters['M_GTG'] = Master_Model(BBone, get_module('GTG', (gtg_module_params, {**ll_module_params, 'module_out': 1})), dataset_name)
        elif not only_bb:
            Masters['M_None'] = Master_Model(BBone, None, dataset_name)
            only_bb = True
        else: continue
    
    # now the master: backbone + module initial checkpoint have been saved 
    
    return Masters
    