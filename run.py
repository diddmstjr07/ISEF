import argparse
import ast
import os
import pathlib
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from module.src.dataset import make_supervised_data_module
from module.src.params import DataArguments, ModelArguments, TrainingArguments
from module.src.trainer import QwenSFTTrainer
from module.src.train.monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
)
from module.src.train.monkey_patch_vision import replace_qwen2_5_vision
from module.src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
)


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank == "0" or local_rank is None:
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def build_train_args(cfg):
    args = [
        "--model_id", cfg.model_id,
        "--data_path", str(cfg.data_path),
        "--image_folder", str(cfg.image_folder),
        "--output_dir", str(cfg.output_dir),
        "--num_train_epochs", str(cfg.num_train_epochs),
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--learning_rate", str(cfg.learning_rate),
        "--merger_lr", str(cfg.merger_lr),
        "--vision_lr", str(cfg.vision_lr),
        "--weight_decay", str(cfg.weight_decay),
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--lr_scheduler_type", cfg.lr_scheduler_type,
        "--logging_steps", str(cfg.logging_steps),
        "--save_strategy", cfg.save_strategy,
        "--save_steps", str(cfg.save_steps),
        "--save_total_limit", str(cfg.save_total_limit),
        "--dataloader_num_workers", str(cfg.dataloader_num_workers),
        "--remove_unused_columns", "False",
        "--freeze_vision_tower", "True",
        "--freeze_llm", "True",
        "--freeze_merger", "False",
        "--use_liger_kernel", str(cfg.use_liger_kernel),
        "--bf16", str(cfg.bf16),
        "--fp16", str(cfg.fp16),
        "--disable_flash_attn2", str(cfg.disable_flash_attn2),
        "--gradient_checkpointing", str(cfg.gradient_checkpointing),
        "--report_to", cfg.report_to,
        "--lazy_preprocess", "True",
    ]

    if cfg.fps is not None:
        args += ["--fps", str(cfg.fps)]
    if cfg.nframes is not None:
        args += ["--nframes", str(cfg.nframes)]
    if cfg.image_min_pixels is not None:
        args += ["--image_min_pixels", str(cfg.image_min_pixels)]
    if cfg.image_max_pixels is not None:
        args += ["--image_max_pixels", str(cfg.image_max_pixels)]
    if cfg.video_min_pixels is not None:
        args += ["--video_min_pixels", str(cfg.video_min_pixels)]
    if cfg.video_max_pixels is not None:
        args += ["--video_max_pixels", str(cfg.video_max_pixels)]
    if cfg.image_resized_width is not None and cfg.image_resized_height is not None:
        args += ["--image_resized_width", str(cfg.image_resized_width)]
        args += ["--image_resized_height", str(cfg.image_resized_height)]
    if cfg.video_resized_width is not None and cfg.video_resized_height is not None:
        args += ["--video_resized_width", str(cfg.video_resized_width)]
        args += ["--video_resized_height", str(cfg.video_resized_height)]
    if cfg.deepspeed is not None:
        args += ["--deepspeed", str(cfg.deepspeed)]

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen-VL SFT with projector-only training.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", default="output/projector_sft")

    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--merger-lr", type=float, default=1e-5)
    parser.add_argument("--vision-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", default="steps")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--report-to", default="tensorboard")

    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--nframes", type=int, default=None)

    parser.add_argument("--image-min-pixels", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--image-resized-width", type=int, default=None)
    parser.add_argument("--image-resized-height", type=int, default=None)
    parser.add_argument("--video-resized-width", type=int, default=None)
    parser.add_argument("--video-resized-height", type=int, default=None)

    parser.add_argument("--lora-enable", action="store_true", default=False)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--freeze-merger", action="store_true", default=False)

    parser.add_argument("--use-liger-kernel", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--disable-flash-attn2", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)

    parser.add_argument("--deepspeed", default=None, help="Optional deepspeed config path.")
    return parser.parse_args()


def train_projector_sft():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, (
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        )
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args,
        )
    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args,
        )
    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args,
        )
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        model.enable_input_require_grads()

    if training_args.bits in [4, 8]:
        model.config.dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_token" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

def main():
    cfg = parse_args()
    data_path = Path(cfg.data_path)
    image_folder = Path(cfg.image_folder)
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")
    if not image_folder.exists():
        raise FileNotFoundError(f"image_folder not found: {image_folder}")

    train_args = build_train_args(cfg)
    sys.argv = ["train"] + train_args
    train_projector_sft()


if __name__ == "__main__":
    main()
