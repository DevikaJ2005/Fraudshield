"""Training entrypoint for FraudShield Colab-friendly experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset

from config import ExperimentConfig
from environment import FraudShieldTextEnvironment
from models import ActionTypeEnum, FraudCheckAction
from utils import ensure_dir, save_json, seed_everything

class ExpertCurriculumTeacher:
    """Teacher policy that uses hidden task structure to generate stronger trajectories."""

    def decide(self, text_env: FraudShieldTextEnvironment) -> FraudCheckAction:
        observation = text_env.current_observation
        case_id = observation.case_id
        revealed = observation.revealed_evidence
        case = text_env.env.workflow_cases[case_id]
        budget = int(observation.app_context.get("investigation_budget_remaining", 0))

        if "transaction_review" not in revealed:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.REVIEW_TRANSACTION,
                reasoning="Open the transaction details before taking any deeper investigative step.",
            )

        planned_sequence = self._planned_evidence_sequence(case)
        for evidence_key, action_type, reasoning in planned_sequence:
            if evidence_key not in revealed and budget > 0:
                return FraudCheckAction(case_id=case_id, action_type=action_type, reasoning=reasoning)

        if observation.note_required:
            return FraudCheckAction(
                case_id=case_id,
                action_type=ActionTypeEnum.ADD_CASE_NOTE,
                note_text=self._case_note(case),
            )

        return FraudCheckAction(
            case_id=case_id,
            action_type=ActionTypeEnum.RESOLVE_CASE,
            resolution=case["correct_resolution"],
            reasoning=self._resolution_reasoning(case),
        )

    def _planned_evidence_sequence(self, case: dict[str, Any]) -> list[tuple[str, ActionTypeEnum, str]]:
        role = case["role"]
        task_specific = [
            (
                "customer_profile",
                ActionTypeEnum.FETCH_CUSTOMER_PROFILE,
                "Customer history is needed to understand whether this pattern reflects risky buyer behavior.",
            ),
            (
                "merchant_profile",
                ActionTypeEnum.FETCH_MERCHANT_PROFILE,
                "Merchant health helps explain whether the case risk comes from the seller side.",
            ),
            (
                "network_graph",
                ActionTypeEnum.FETCH_NETWORK_GRAPH,
                "Linked-activity evidence is needed to confirm whether this case participates in a broader cluster.",
            ),
            (
                "policy_guide",
                ActionTypeEnum.CHECK_POLICY,
                "Policy guidance is required before choosing the final route.",
            ),
        ]

        if role == "single" and case["correct_resolution"].value == "request_docs":
            return [
                task_specific[0],
                task_specific[3],
                task_specific[1],
            ]
        if role == "primary":
            return [
                task_specific[2],
                task_specific[1],
                task_specific[3],
            ]
        if role == "secondary":
            return [
                task_specific[2],
                task_specific[0],
                task_specific[3],
            ]
        return [
            task_specific[1],
        ]

    def _case_note(self, case: dict[str, Any]) -> str:
        if case["role"] == "primary":
            return "Reviewed the transaction trace, graph evidence, merchant signals, and policy guidance before escalating the linked primary case."
        if case["role"] == "secondary":
            return "Reviewed the transaction trace, graph evidence, customer history, and policy guidance before finalizing the linked secondary case."
        if case["correct_resolution"].value == "request_docs":
            return "Reviewed transaction, customer, merchant, and policy evidence before requesting more supporting documents."
        return "Reviewed the transaction evidence and documented the case before final routing."

    def _resolution_reasoning(self, case: dict[str, Any]) -> str:
        mapping = {
            "approve": "The collected evidence supports approval without additional intervention.",
            "block": "The combined evidence supports blocking the transaction as high risk.",
            "hold": "The evidence remains risky enough to hold the case for more controlled handling.",
            "request_docs": "The case is ambiguous enough that supporting documents are the safest next step.",
            "escalate": "The linked-cluster evidence and loss risk justify escalation to a higher-touch reviewer.",
        }
        return mapping[case["correct_resolution"].value]


def build_public_curriculum(config: ExperimentConfig) -> Dataset:
    """Load public fraud examples and convert them into action-centric prompts."""

    dataset_name = config.training.public_curriculum_dataset
    dataset = load_dataset(dataset_name, split="train")
    rows: list[dict[str, Any]] = []
    for row in dataset.shuffle(seed=config.seed).select(
        range(min(config.training.public_curriculum_rows, len(dataset)))
    ):
        amount = float(row.get("amount", row.get("Amount", 0.0)) or 0.0)
        label = int(row.get("is_fraud", row.get("isFraud", row.get("Class", 0))) or 0)
        transaction_type = str(row.get("transaction_type", row.get("type", "purchase")))
        prompt = (
            "You are a fraud analyst learning to investigate risk before final routing. Return JSON only.\n\n"
            f"Visible observation:\n{json.dumps({'amount_usd': amount, 'transaction_type': transaction_type, 'task_name': 'medium', 'available_investigations': ['merchant_profile', 'customer_profile', 'network_graph', 'payment_trace', 'policy_review']})}\n\n"
            'JSON schema: {"action_type":"investigate|decide","investigation_target":"alias_or_null","decision":"fraud|legitimate|null","confidence":0.0,"reasoning":"one sentence"}'
        )
        if label:
            payload = {
                "action_type": "investigate",
                "investigation_target": "network_graph" if amount > 1000 else "payment_trace",
                "decision": None,
                "confidence": None,
                "reasoning": "The visible transaction is risky, so gather stronger network or payment evidence first.",
            }
        else:
            payload = {
                "action_type": "decide",
                "investigation_target": None,
                "decision": "legitimate",
                "confidence": 0.8,
                "reasoning": "The visible transaction appears low risk and can be cleared confidently.",
            }
        rows.append({"text": prompt + "\n" + json.dumps(payload, separators=(",", ":")), "source": "public"})
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def build_rollout_dataset(config: ExperimentConfig) -> Dataset:
    """Generate environment-compatible trajectories from an expert teacher."""

    text_env = FraudShieldTextEnvironment(config.environment, config.reward_weights)
    agent = ExpertCurriculumTeacher()
    rows: list[dict[str, Any]] = []
    for task_name in config.evaluation.tasks:
        for _ in range(config.training.warmstart_rollouts_per_task):
            prompt = text_env.reset(task=task_name)
            done = False
            while not done:
                action = agent.decide(text_env)
                payload = {
                    "action_type": "decide" if action.action_type.value == "resolve_case" else "investigate",
                    "investigation_target": action.action_type.value,
                    "decision": "fraud" if getattr(action, "resolution", None) and action.resolution.value in {"block", "hold", "escalate"} else "legitimate",
                    "confidence": 0.8,
                    "reasoning": action.reasoning or "Training rollout step.",
                }
                rows.append({"text": prompt + "\n" + json.dumps(payload, separators=(",", ":")), "source": "rollout"})
                step = text_env.step(json.dumps(payload))
                prompt = step.next_prompt
                done = step.done
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def load_model_stack(config: ExperimentConfig):
    """Load a Colab-friendly 4-bit LoRA stack."""

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base_model,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=config.model.gradient_checkpointing,
    )
    return model, tokenizer


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    """Run the configured training pipeline."""

    seed_everything(config.seed)
    ensure_dir(config.training.output_dir)
    ensure_dir(config.training.checkpoint_dir)
    if "wandb" in config.training.report_to and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = "fraudshield"
    if "tensorboard" in config.training.report_to:
        ensure_dir(Path(config.training.output_dir) / "tb_logs")
    public_dataset = build_public_curriculum(config)
    rollout_dataset = build_rollout_dataset(config)
    model, tokenizer = load_model_stack(config)

    from transformers import TrainingArguments
    from trl import SFTTrainer

    stage1_args = TrainingArguments(
        output_dir=str(Path(config.training.output_dir) / "stage1"),
        num_train_epochs=1,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate * 2,
        logging_steps=max(1, config.training.logging_steps),
        save_strategy="no",
        report_to=config.training.report_to,
        run_name=f"{config.training.run_name}-stage1",
        logging_dir=str(Path(config.training.output_dir) / "tb_logs" / "stage1"),
    )
    stage1_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=public_dataset,
        dataset_text_field="text",
        max_seq_length=config.model.max_seq_length,
        packing=False,
        args=stage1_args,
    )
    stage1_trainer.train()

    stage2_args = TrainingArguments(
        output_dir=str(Path(config.training.output_dir) / "stage2"),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        logging_steps=max(1, config.training.logging_steps),
        save_strategy="epoch",
        report_to=config.training.report_to,
        run_name=f"{config.training.run_name}-stage2",
        logging_dir=str(Path(config.training.output_dir) / "tb_logs" / "stage2"),
    )
    trainer = SFTTrainer(
        model=stage1_trainer.model,
        tokenizer=tokenizer,
        train_dataset=rollout_dataset,
        dataset_text_field="text",
        max_seq_length=config.model.max_seq_length,
        packing=False,
        args=stage2_args,
    )
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    output_dir = Path(config.training.output_dir) / "trained_policy"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_history = trainer.state.log_history
    loss_points = [(entry["step"], entry["loss"]) for entry in log_history if "step" in entry and "loss" in entry]
    if loss_points:
        xs, ys = zip(*loss_points)
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.xlabel("training step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(Path(config.training.output_dir) / "loss_vs_steps.png")
        plt.close()

    reward_trace = []
    for idx, entry in enumerate(log_history, start=1):
        if "loss" in entry:
            reward_trace.append(max(0.0, 1.0 - float(entry["loss"])))
    if reward_trace:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(reward_trace) + 1), reward_trace, label="reward_proxy")
        window = min(10, len(reward_trace))
        if window:
            from utils import moving_average

            plt.plot(range(1, len(reward_trace) + 1), moving_average(reward_trace, window=window), label="moving_avg")
        plt.xlabel("training step")
        plt.ylabel("reward proxy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(config.training.output_dir) / "reward_vs_steps.png")
        plt.close()

    metadata = {
        "status": "completed",
        "algorithm": config.training.algorithm,
        "warmstart_algorithm": config.training.warmstart_algorithm,
        "report_to": config.training.report_to,
        "run_name": config.training.run_name,
        "public_curriculum_dataset": config.training.public_curriculum_dataset,
        "output_dir": str(output_dir),
        "num_public_examples": len(public_dataset),
        "num_rollout_examples": len(rollout_dataset),
        "log_history": log_history,
    }
    save_json(metadata, Path(config.training.output_dir) / "training_run_summary.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FraudShield with a Colab-friendly curriculum.")
    parser.add_argument("--config", default="configs/colab_qlora_grpo.json", help="Path to experiment config JSON.")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    config.save(Path(config.training.output_dir) / "resolved_config.json")
    summary = run_training(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
