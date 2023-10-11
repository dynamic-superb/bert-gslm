import argparse
from pathlib import Path

from fairseq import options
from fairseq_cli.sample import cli_main


# storage_dir = utils.get_storage_dir()


def convert_to_fairseq_args(args):
    fairseq_parser = options.get_interactive_generation_parser()
    fairseq_parser.add_argument("--input_src", type=str, default=None, required=True)
    fairseq_parser.add_argument("--output", type=str, default=None)
    fairseq_parser.add_argument("--prompt_path", type=str, default=None)
    fairseq_parser.add_argument("--raw_file", type=str, default=None)
    fairseq_parser.add_argument("--label_dir", type=str, default=None)

    # not useful
    fairseq_parser.add_argument("--samples_per_prompt", type=int, default=1)
    fairseq_parser.add_argument("--debug", action="store_true")
    fairseq_parser.add_argument("--slice_sampling", action="store_true")
    # fairseq_parser.add_argument("--without_beam_search", action="store_true", default=False)

    """
    --input_src: path to the test split
    --output: path to the output file
    --prompt_path: path to the prompt model
    --raw_file: path to the reduced units file, which provides file names for each utterance (users don't need to pass this arg)
    """
    args.save_dir.mkdir(parents=True, exist_ok=True)
    input_src = args.data_dir / "test.txt"
    output = args.save_dir / "samples.json"
    prompt_path = args.model_dir / "checkpoint_best.pt"
    # prompt_path = None
    base_model_path = args.model_dir / "checkpoint_best.pt"
    raw_file = args.data_dir / "test_files.txt"

    speech_prompt_args = [
        f"--input_src={input_src}",
        f"--output={output}",
        # f"--prompt_path={prompt_path}",
        f"--raw_file={raw_file}",
        f"--label_dir={args.label_dir}",
    ]
    
    path_args = [
        str(args.data_dir),
        f"--path={base_model_path}",
    ]

    sampling_args = [
        "--user-dir=fairseq_usr",
        "--task=language_modeling",
        "--sampling",
        "--sampling-topk=1",
        f"--seed={args.seed}",
        "--max-len-a=0",
        "--max-len-b=150",
        "--num-workers=12",
        "--prefix-size=-21",
        "--skip-invalid-size-inputs-valid-test",
        f"--batch-size={args.batch_size}",
    ]
    # if args.without_beam_search:
    #     sampling_args += ["--without-beam-search"]
    if args.fp16:
        sampling_args += ["--fp16"]

    fairseq_args = path_args + sampling_args + speech_prompt_args
    print(fairseq_args)
    return fairseq_parser, fairseq_args


def get_input_args():
    # ========== #
    #   Parser   #
    # ========== #
    parser = options.get_interactive_generation_parser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--exp_name", type=str, default="my_exp")
    parser.add_argument(
        "--vb_method", type=str, default="freq", choices=["freq", "random", "learnable"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--without_beam_search", action="store_true", default=False)
    parser.add_argument("--label_dir", type=str, default="/mnt/data/gslm-test-meta-for-stan")


    args = parser.parse_args()
    fairseq_parser, fairseq_args = convert_to_fairseq_args(args)
    return fairseq_parser, fairseq_args


def main():
    fairseq_parser, fairseq_args = get_input_args()
    cli_main(fairseq_parser, fairseq_args)


if __name__ == "__main__":
    main()
