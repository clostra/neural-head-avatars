from nha.optimization.train_pl_module_multivideo import train_pl_module_multivideo
from nha.util.log import get_logger
from nha.data.real import RealDataModule, MultipleVideosDataModule
from nha.models.nha_multiple_video_optimizer import NHAMultipleVideoOptimizer

logger = get_logger("nha", root=True)

if __name__ == "__main__":
    train_pl_module_multivideo(NHAMultipleVideoOptimizer, MultipleVideosDataModule)