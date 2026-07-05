from .config import HFM1DConfig
from .model import HFM1D
from .encoder import FrameEncoder
from .evolution import EvolutionOperator
from .decoder import SlotDecoder
from .context_encoder import ContextEncoder
from .discriminator import HFMDiscriminator
from .trainer import RolloutGANTrainer, train_step_gan, FluidLoss
from .autoencoder import PairEncoder, LatentAutoencoder, AutoencoderTrainer
from .latent_dynamics import LatentDynamics, LatentDynamicsTrainer

__all__ = [
    "HFM1DConfig", "HFM1D",
    "FrameEncoder", "EvolutionOperator", "SlotDecoder",
    "ContextEncoder", "HFMDiscriminator",
    "RolloutGANTrainer", "train_step_gan", "FluidLoss",
    # BPTT-free two-phase design
    "PairEncoder", "LatentAutoencoder", "AutoencoderTrainer",
    "LatentDynamics", "LatentDynamicsTrainer",
]
