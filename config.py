"""Config objects for FraudShield RL-style experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils import load_json, save_json


@dataclass
class RewardWeights:
    """Weights used to combine decomposed reward subscores."""

    env_reward: float = 1.0
    correctness: float = 0.35
    task_completion: float = 0.20
    reasoning_quality: float = 0.10
    efficiency: float = 0.10
    safety: float = 0.10
    formatting_compliance: float = 0.10
    consistency: float = 0.05


@dataclass
class EnvironmentConfig:
    """Environment-facing configuration."""

    data_path: str = "data"
    default_task: str = "medium"
    max_rollout_steps: int = 14
    seed: int = 42


@dataclass
class ModelConfig:
    """Model and adapter configuration for Colab-friendly training."""

    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    gradient_checkpointing: str = "unsloth"
    mixed_precision: str = "auto"


@dataclass
class TrainingConfig:
    """Trainer, rollout, and checkpoint parameters."""

    algorithm: str = "grpo"
    warmstart_algorithm: str = "sft"
    output_dir: str = "artifacts/rl_runs/default"
    checkpoint_dir: str = "artifacts/rl_runs/default/checkpoints"
    save_to_drive: bool = False
    drive_dir: str = "/content/drive/MyDrive/fraudshield"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    eval_every_steps: int = 10
    save_every_steps: int = 20
    warmstart_rollouts_per_task: int = 24
    rl_rollouts_per_task: int = 8
    max_prompt_tokens: int = 2048
    max_completion_tokens: int = 220
    logging_steps: int = 1
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str = "fraudshield-colab-run"
    resume_from_checkpoint: str | None = None
    public_curriculum_dataset: str = "Phoenix21/mock_fraud-detection-dataset"
    public_curriculum_rows: int = 2500


@dataclass
class EvaluationConfig:
    """Evaluation and plotting configuration."""

    tasks: list[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    fixed_prompt_cases: int = 3
    plots_dir: str = "artifacts/plots"
    compare_against_base_model: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "fraudshield-colab-qlora-grpo"
    seed: int = 42
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    reward_version: str = "v1"
    ablation_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=data.get("name", cls().name),
            seed=data.get("seed", cls().seed),
            environment=EnvironmentConfig(**data.get("environment", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            reward_weights=RewardWeights(**data.get("reward_weights", {})),
            reward_version=data.get("reward_version", "v1"),
            ablation_tags=list(data.get("ablation_tags", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(load_json(path))
