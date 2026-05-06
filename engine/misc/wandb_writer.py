"""
Wrapper over TensorBoard SummaryWriter that mirrors metrics to Weights & Biases.
Drop-in replacement: supports the same add_scalar / add_text interface used by
the RT-DETRv4 training loop, while also calling wandb.log() transparently.
"""

import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

_logger = logging.getLogger(__name__)


class WandbWriter(SummaryWriter):
    """
    SummaryWriter subclass that additionally logs scalars to wandb.

    Parameters
    ----------
    log_dir   : TensorBoard log directory (passed to SummaryWriter)
    project   : wandb project name
    name      : wandb run name (defaults to the last component of log_dir)
    config    : dict of hyper-parameters to attach to the wandb run
    """

    def __init__(self, log_dir: str, project: str, name: str = None, config: dict = None, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Run:  pip install wandb"
            )

        run_name = name or Path(log_dir).name
        _logger.info(f"[WandbWriter] Initialising wandb run '{run_name}' in project '{project}'")

        self._wandb = wandb
        self._wandb_run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            resume="allow",
            dir=log_dir,
        )
        # Buffer: collect scalars logged in one global_step so we can send them
        # as a single wandb.log() call.
        self._step_buffer: dict = {}
        self._current_step: int = -1

    # ------------------------------------------------------------------
    # SummaryWriter overrides
    # ------------------------------------------------------------------

    def add_scalar(self, tag: str, scalar_value, global_step: int = None, **kwargs):
        # Write to TensorBoard as usual
        super().add_scalar(tag, scalar_value, global_step=global_step, **kwargs)

        # Accumulate for wandb
        if global_step != self._current_step:
            self._flush_wandb()
            self._current_step = global_step

        # Replace '/' with '_' so wandb groups nicely, keep the original tag too
        wandb_key = tag.replace("/", "_")
        self._step_buffer[wandb_key] = scalar_value

    def add_text(self, tag: str, text_string: str, global_step: int = None, **kwargs):
        super().add_text(tag, text_string, global_step=global_step, **kwargs)
        # Log once as a wandb summary entry (not repeated every step)
        if self._wandb_run is not None:
            self._wandb_run.summary[tag] = text_string

    def close(self):
        self._flush_wandb()
        super().close()
        if self._wandb_run is not None:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_wandb(self):
        if self._step_buffer and self._wandb_run is not None:
            self._wandb.log(self._step_buffer, step=self._current_step)
            self._step_buffer = {}
