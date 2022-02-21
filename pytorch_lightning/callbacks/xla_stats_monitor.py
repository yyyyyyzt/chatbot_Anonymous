# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
XLA Stats Monitor
=================

Monitor and logs XLA stats during training.

"""
import time

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import _TPU_AVAILABLE, DeviceType, rank_zero_deprecation, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class XLAStatsMonitor(Callback):
    r"""
    .. deprecated:: v1.5
        The `XLAStatsMonitor` callback was deprecated in v1.5 and will be removed in v1.7.
        Please use the `DeviceStatsMonitor` callback instead.

    Automatically monitors and logs XLA stats during training stage. ``XLAStatsMonitor`` is a callback and in
    order to use it you need to assign a logger in the ``Trainer``.

    Args:
        verbose: Set to ``True`` to print average peak and free memory, and epoch time
            every epoch.

    Raises:
        MisconfigurationException:
            If not running on TPUs, or ``Trainer`` has no logger.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import XLAStatsMonitor
        >>> xla_stats = XLAStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[xla_stats]) # doctest: +SKIP
    """

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()

        rank_zero_deprecation(
            "The `XLAStatsMonitor` callback was deprecated in v1.5 and will be removed in v1.7."
            " Please use the `DeviceStatsMonitor` callback instead."
        )

        if not _TPU_AVAILABLE:
            raise MisconfigurationException("Cannot use XLAStatsMonitor with TPUs are not available")

        self._verbose = verbose

    def on_train_start(self, trainer, pl_module) -> None:
        if not trainer.logger:
            raise MisconfigurationException("Cannot use XLAStatsMonitor callback with Trainer that has no logger.")

        if trainer._device_type != DeviceType.TPU:
            raise MisconfigurationException(
                "You are using XLAStatsMonitor but are not running on TPU"
                f" since `tpu_cores` attribute in Trainer is set to {trainer.tpu_cores}."
            )

        memory_info = xm.get_memory_info(pl_module.device)
        total_memory = trainer.training_type_plugin.reduce(memory_info["kb_total"]) * 0.001
        rank_zero_info(f"Average Total memory: {total_memory:.2f} MB")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        logs = {}
        memory_info = xm.get_memory_info(pl_module.device)
        epoch_time = time.time() - self._start_time

        free_memory = memory_info["kb_free"]
        peak_memory = memory_info["kb_total"] - free_memory

        free_memory = trainer.training_type_plugin.reduce(free_memory) * 0.001
        peak_memory = trainer.training_type_plugin.reduce(peak_memory) * 0.001
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)

        logs["avg. free memory (MB)"] = free_memory
        logs["avg. peak memory (MB)"] = peak_memory
        trainer.logger.log_metrics(logs, step=trainer.current_epoch)

        if self._verbose:
            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory: {peak_memory:.2f} MB")
            rank_zero_info(f"Average Free memory: {free_memory:.2f} MB")
