# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Phi-3/Phi-4 AutoDeploy model config patches.

Verifies the fused-projection tp_plan override: HF's compound TP policies
(``colwise_gather_output`` / ``rowwise_split_input``) are rewritten to their
pure ``colwise`` / ``rowwise`` forms for Phi configs only, without mutating the
shared class-level default dict. See models/patches/phi.py and issue #14679.
"""

from transformers import AutoConfig

from tensorrt_llm._torch.auto_deploy.models.patches.phi import _override_phi_fused_tp_plan


def test_phi_fused_tp_plan_override_rewrites_compound_policies():
    config = AutoConfig.for_model("phi3")

    # The real Phi-3/Phi-4 plan declares the fused projections with HF's compound
    # gather/split policies (Phi-4 reuses the Phi-3 config, model_type == "phi3").
    assert any(v == "colwise_gather_output" for v in config.base_model_tp_plan.values())
    assert any(v == "rowwise_split_input" for v in config.base_model_tp_plan.values())
    class_default = dict(type(config).base_model_tp_plan)

    _override_phi_fused_tp_plan(config)

    # Every fused entry is now a pure colwise/rowwise policy.
    assert set(config.base_model_tp_plan.values()) <= {"colwise", "rowwise"}
    assert config.base_model_tp_plan["layers.*.self_attn.qkv_proj"] == "colwise"
    assert config.base_model_tp_plan["layers.*.self_attn.o_proj"] == "rowwise"
    assert config.base_model_tp_plan["layers.*.mlp.gate_up_proj"] == "colwise"
    assert config.base_model_tp_plan["layers.*.mlp.down_proj"] == "rowwise"

    # The shared class-level default must NOT have been mutated.
    assert type(config).base_model_tp_plan == class_default


def test_phi_tp_plan_override_skips_non_phi():
    class _Other:
        model_type = "llama"
        base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise_gather_output"}

    other = _Other()
    _override_phi_fused_tp_plan(other)
    assert other.base_model_tp_plan == {"layers.*.self_attn.q_proj": "colwise_gather_output"}
