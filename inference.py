from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.MODEL.WEIGHTS(os.path.join(cfg.OUT))