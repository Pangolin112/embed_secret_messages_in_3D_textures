# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Code to train model, only needed in order to not save InstructPix2Pix checkpoints
"""
from dataclasses import dataclass, field
from typing import Type
import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.decorators import check_main_thread

@dataclass
class DNTrainerConfig(TrainerConfig):
    """Configuration for the InstructGS2GSTrainer."""
    _target: Type = field(default_factory=lambda: DNTrainer)


class DNTrainer(Trainer):
    """Trainer for InstructGS2GS"""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        # drop anything under ip2p_depth and ip2p_ptd, or the saved checkpoint will be too large
        # pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "ip2p." not in k}
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if not (k.startswith("lseg_model.") or k.startswith("sam2_predictor.") or k.startswith("ip2p_ptd.") or k.startswith("ip2p_depth.") or k.startswith("MODEL.") or k.startswith("CONDITIONER.") or k.startswith("AE.") or k.startswith("DENOISER."))}
        
        # since the optimizer for normal is just to make the trainer happy, so
        # we need to mannually add the 'step' dict into it.
        optimizer_states = {}
        for name, opt in self.optimizers.optimizers.items():
            sd = opt.state_dict()

            if name == "normals":
                sd = opt.state_dict()
                state = sd["state"]

                # build a map from pid → the actual parameter Tensor
                id_to_param = {
                    id(p): p
                    for group in opt.param_groups
                    for p in group["params"]
                }

                # iterate only over the pids already in state, and skip any we can't find
                for pid, st in list(state.items()):
                    # 1) force the same global step
                    st.setdefault("step", step)
        
                    # look up the real tensor
                    param = id_to_param.get(pid, None)
                    if param is None:
                        # no matching parameter in param_groups → skip buffer creation
                        continue
        
                    # 2) only add exp_avg if missing—and zero it with the correct shape
                    st.setdefault("exp_avg", torch.zeros_like(param))
                    # 3) only add exp_avg_sq if missing
                    st.setdefault("exp_avg_sq", torch.zeros_like(param))

            # everybody else, leave exactly as they are
            optimizer_states[name] = sd
        
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                # "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "optimizers": optimizer_states,
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )

        # comment this for early and late stage training effects comparison
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # # delete everything else in the checkpoint folder
            # for f in self.checkpoint_dir.glob("*"):
            #     if f != ckpt_path:
            #         f.unlink()
            # delete only old .ckpt files (won't touch images/ or other folders)
            for ckpt in self.checkpoint_dir.glob("*.ckpt"):
                if ckpt != ckpt_path:
                    ckpt.unlink()
