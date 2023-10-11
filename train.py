import argparse
import yaml
from pathlib import Path

from fairseq import options
from fairseq_cli.train import cli_main

from data_utils import GlobalConstants


def convert_to_fairseq_args(args):
    fairseq_parser = options.get_training_parser()
    fairseq_parser.add_argument("--prompt_length", type=int)
    fairseq_parser.add_argument("--deep_prompt", action="store_true")
    fairseq_parser.add_argument("--num_classes", type=int, default=0)
    fairseq_parser.add_argument("--linear_verbalizer", action="store_true")
    # fairseq_parser.add_argument("--use_nlp_embedding", action="store_true", default=False)
    # fairseq_parser.add_argument("--nlp_model_name", type=str, default="bert-base-uncased")
    
    config_path = Path("data_utils") / "preprocessors" / args.downstream / "config.yaml"
    with config_path.open(mode="r") as f:
        task_config = yaml.load(f, Loader=yaml.FullLoader)

    # Disable prompt.
    # TODO: Remove prompt-related codes from the codebase.
    speech_prompt_args = [f"--prompt_length={args.prompt_length}"]
    if args.deep_prompt:
        speech_prompt_args += ["--deep_prompt"]

    if task_config["task_type"] == "classification":
        num_classes = task_config["num_classes"]
        speech_prompt_args += [f"--num_classes={num_classes}"]

    if args.linear_verbalizer:
        speech_prompt_args += ["--linear_verbalizer"]
        
    if args.fine_tune:
        speech_prompt_args += ["--fine-tune"]
        
    if args.use_nlp_embedding:
        speech_prompt_args += ["--use-nlp-embedding", f"--nlp-model-name={args.nlp_model_name}"]
    
    if args.freeze_nlp_model:
        speech_prompt_args += ["--freeze-nlp-model"]
        
    # ====================== #
    #   Fairseq-train args   #
    # ====================== #
    # For Fairseq-train args, please refer to the official documentation.
    # https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train

    restore_file = GlobalConstants.SAVE_ROOT / "HuBERT" / "checkpoint_best.pt"
    tensorboard_dir = args.save_dir / "logs"
    ckpt_dir = args.save_dir / "ckpts"

    path_args = [
        str(args.data_dir),
        f"--restore-file={restore_file}",
        f"--tensorboard-logdir={tensorboard_dir}",
        f"--save-dir={ckpt_dir}",
        f"--keep-last-epochs={args.keep_last_epochs}"
        
    ]

    task_args = [
        "--user-dir=fairseq_usr",
        "--task=prompt_language_modeling",
        "--arch=GSLM_SpeechPrompt_v1",
        "--criterion=cross_entropy_prompt",
        "--share-decoder-input-output-embed",
        "--sample-break-mode=eos",
    ]

    training_args = [
        "--ddp-backend=legacy_ddp",
        "--reset-optimizer",
        f"--batch-size={args.batch_size}",
        f"--lr={args.learning_rate}",
        "--optimizer=adam",
        "--adam-betas=(0.9, 0.98)",
        f"--clip-norm={args.clip_grad_norm}",
        "--update-freq=1",
        f"--max-tokens={args.max_tokens}",
        f"--num-workers={args.num_workers}",
        "--skip-invalid-size-inputs-valid-test",
        f"--patience={args.patience}",
        f"--max-epoch={args.max_epoch}",
        f"--log-interval={args.log_interval}",
        f"--seed={args.seed}",
        f"--dropout={args.dropout}",
        f"--attention-dropout={args.attn_dropout}",
        f"--save-interval={args.save_interval}",
    ]

    if args.fp16:
        training_args += ["--fp16"]

    fairseq_args = path_args + task_args + training_args + speech_prompt_args
    print(fairseq_args)
    return fairseq_parser, fairseq_args


def get_input_args():
    # ========== #
    #   Parser   #
    # ========== #
    parser = argparse.ArgumentParser()
    parser.add_argument("downstream", type=str)
    parser.add_argument("--exp_name", type=str, default="my_exp")
    parser.add_argument("--data_dir", type=Path, default=Path("."))
    parser.add_argument("--save_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--vb_method", type=str, choices=["random", "freq", "learnable"]
    )
    parser.add_argument("--linear_verbalizer", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--keep_last_epochs", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=False)
    
    # Prompt
    parser.add_argument("--prompt_length", type=int, default=0)
    parser.add_argument("--deep_prompt", action="store_true", default=False)
    parser.add_argument("--fine_tune", action="store_true", default=True)
    
    # Bert
    parser.add_argument("--use_nlp_embedding", action="store_true", default=False)
    parser.add_argument("--nlp_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_nlp_model", action="store_true", default=True)
    args = parser.parse_args()

    fairseq_parser, fairseq_args = convert_to_fairseq_args(args)

    return fairseq_parser, fairseq_args

def main():
    fairseq_parser, fairseq_args = get_input_args()
    cli_main(fairseq_parser, fairseq_args)


if __name__ == "__main__":
    main()
