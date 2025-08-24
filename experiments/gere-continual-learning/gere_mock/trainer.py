from transformers import Trainer
import logging

logger = logging.getLogger(__name__)


class GeReTrainer(Trainer):
    """
    A mock implementation of the GeReTrainer for integration testing.
    This class accepts the GeRe-specific arguments and logs them, but
    delegates the actual training to the standard Hugging Face Trainer.
    This allows for testing the integration pattern without needing the
    original GeRe source code.
    """

    def __init__(
        self,
        *args,
        gere_hidden_state_saving_dir: str = None,
        reuse_gere_hidden_state: bool = False,
        num_interpolate_per_batch: int = 0,
        w_strategy: str = "dy",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gere_hidden_state_saving_dir = gere_hidden_state_saving_dir
        self.reuse_gere_hidden_state = reuse_gere_hidden_state
        self.num_interpolate_per_batch = num_interpolate_per_batch
        self.w_strategy = w_strategy

        logger.warning("--- Using Mock GeReTrainer ---")
        logger.info(f"GeRe integration successful. Received parameters:")
        logger.info(
            f"  gere_hidden_state_saving_dir: {self.gere_hidden_state_saving_dir}"
        )
        logger.info(f"  reuse_gere_hidden_state: {self.reuse_gere_hidden_state}")
        logger.info(f"  num_interpolate_per_batch: {self.num_interpolate_per_batch}")
        logger.info(f"  w_strategy: {self.w_strategy}")
        logger.warning("Mock trainer will now proceed with standard training loop.")

    def train(self, *args, **kwargs):
        # In a real implementation, this is where the general replay logic
        # would be executed. Here, we just log and proceed.
        logger.info("Mock GeReTrainer: Entering training loop.")
        result = super().train(*args, **kwargs)
        logger.info("Mock GeReTrainer: Exiting training loop.")
        return result
