#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Sample from a trained LM; hacked fairseq-interactive
"""
import ast
import json
import os
import random
from collections import namedtuple
from email.policy import default
from pathlib import Path
from unittest import result

import numpy as np
import torch
import tqdm
from fairseq import checkpoint_utils, options, tasks, utils
import yaml

Batch = namedtuple("Batch", "ids src_tokens src_lengths")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

global_config_path = Path(__file__).parent.parent.resolve() / "config.yaml"
with open(global_config_path, "r") as file:
    global_config = yaml.safe_load(file)


def slice_by_length(data, max_len=400):
    def slice(src, src_len, max_len=400):
        # === get n_splice === #
        n_slice = 1
        l = src_len
        while l > max_len:
            n_slice *= 2
            l /= 2
        l = int(l)
        # === slice === #
        result = []
        s = src.split()
        for i in range(n_slice):
            s_tokens = s[i * l : (i + 1) * l]
            if s_tokens[-1] != "<s>":
                s_tokens.append("<s>")
            s_str = " ".join(s_tokens)
            result.append(s_str)
        return result

    for i, d in enumerate(data):
        if d["src_len"] > max_len:
            src = d["src"]
            srcs = slice(src, d["src_len"], max_len)  # function
            for i, s in enumerate(srcs):
                a = {
                    "id": f'{d["id"]}_{i}',
                    "file_name": d["file_name"],
                    "src": s,
                    "label": d["label"],
                    "src_len": len(s[i].split()),
                    "label_len": d["label_len"],
                }
                data.append(a)


def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.dataset.max_tokens,
        max_sentences=args.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch["id"],
            src_tokens=batch["net_input"]["src_tokens"],
            src_lengths=batch["net_input"]["src_lengths"],
        )

def pad(l, n):
    return l[:n] + [2]*(n-len(l))
def main(args):
    arg_input = args.input_src
    arg_slice_sampling = args.slice_sampling
    arg_output = args.output
    arg_raw_file = args.raw_file
    arg_debug = args.debug
    arg_sample_size = args.samples_per_prompt
    arg_prompt_path = args.prompt_path
    label_dir = args.label_dir
    # without beam_search
    # Should we 
    without_beam_search =  True 
    try:
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf

        args = convert_namespace_to_omegaconf(args)
    except:
        pass

    # if args.max_tokens is None and args.max_sentences is None:
    if args.common.seed is not None:
        np.random.seed(args.common.seed)
        utils.set_torch_seed(args.common.seed)

    if args.generation.sampling:
        args.generation.nbest = args.generation.beam = arg_sample_size

    task = tasks.setup_task(args.task)

    overrides = ast.literal_eval(args.common_eval.model_overrides)

    # load base prompt model
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.common_eval.path.split(os.pathsep),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    model_dict = models[0].state_dict()

    # load prompt and other filtered parameters
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    if arg_prompt_path is not None:
        prompt_dict = torch.load(arg_prompt_path)["model"]
        assert (
            len(prompt_dict.keys()) < 100
        ), "Prompt checkpoint is too large. Please check. If you fine-tuned the whole model, please set --prompt_path to None."
        param_filters = global_config["prompt_param_filter"]
        for name, param in model_dict.items():
            for filterd_name in param_filters:
                if filterd_name in name:
                    model_dict[name] = prompt_dict[name]
        models[0].load_state_dict(model_dict)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    # ========== #
    output_path = Path(arg_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {}
    output_file = output_path

    data = []
    with open(arg_input, "r") as f:
        lines = f.readlines()

    with open(arg_raw_file, "r") as f:
        file_names = [line.strip() for line in f.readlines()]
    
    if label_dir is not None and label_dir != '':
        print("Force label !!!")
        force_label = True
        with open(os.path.join(label_dir, arg_raw_file.split('/')[-2] + '.txt'), 'r') as f:
            label_table = {}
            for i, line in enumerate(f.readlines()):
                data = [l.strip() for l in line.split(',') if l != '']
                label_table[file_names[i]] = []
                for d in data[1:]:
                    # if d.split() > 0:
                    label_table[file_names[i]].append([int(token) for token in d.split()]+[src_dict.eos()])   
                max_len = max([len(l) for l in label_table[file_names[i]]])
                label_table[file_names[i]] = [pad(l, max_len) for l in label_table[file_names[i]]]
    else:
        force_label = False
    data = []
    split = [x.split("<s>") for x in lines]
    srcs = [x[0] + "<s>" for x in split]
    labels = [x[1].strip() for x in split]
    for i in range(len(file_names)):
        data.append(
            {
                "id": i,
                "file_name": file_names[i],
                "src": srcs[i],
                "label": labels[i],
                "src_len": len(srcs[i].split()),
                "label_len": len(labels[i].split()),
            }
        )
    if arg_slice_sampling:
        slice_by_length(data)
    # data = sorted(data, key=lambda d: d["src_len"], reverse=True)[100:200]
    prompts = [d["src"] for d in data]
    
    
    generator = task.build_generator(models, args.generation)
    start_id = 0
    pbar = tqdm.tqdm(total=len(data))

    if models[0].cfg.linear_verbalizer:
        args.dataset.batch_size = 1

    for batch in make_batches(prompts, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }

        results = []
        if without_beam_search:
            models[0].eval()
            src_len = sample["net_input"]["src_tokens"].size()[1]
            max_len = int(
                args.generation.max_len_a * int(src_len) + args.generation.max_len_b
            )

            with torch.no_grad():
                prediction = []
                all_tokens = np.arange(30626)
                            
                for step in range(max_len):
                    output, _ = models[0](**sample["net_input"])
                    pred = output[:, -1, :]
                    if force_label:
                        label = np.array(label_table[data[start_id]["file_name"]])
                        not_allow_tokens = torch.from_numpy(np.delete(all_tokens.copy(), label[:,step]+3)).to(pred.device)

                        pred = pred.index_fill(1, not_allow_tokens, float('-inf'))
                    pred = np.argmax(pred.cpu().data.numpy()) 
                    if force_label:
                        label_table[data[start_id]["file_name"]] = [tokens for tokens in label_table[data[start_id]["file_name"]] if tokens[step] == pred-3]
                        assert len(label_table[data[start_id]["file_name"]]) > 0
                    
                    if (pred == src_dict.eos()) or (force_label and (pred-3) == src_dict.eos()):
                        break

                    src_tokens = sample["net_input"]["src_tokens"]
                    src_tokens = torch.cat(
                        (src_tokens, torch.tensor([[pred]]).cuda()), 1
                    )
                    src_lengths = sample["net_input"]["src_lengths"]
                    src_lengths = src_lengths + 1
                    sample = {
                        "net_input": {
                            "src_tokens": src_tokens.cuda(),
                            "src_lengths": src_lengths.cuda(),
                        },
                    }

                    prediction.append(str(pred - 3))

                data_point = {
                    "file_name": data[start_id]["file_name"],
                    "src": data[start_id]["src"],
                    "label": data[start_id]["label"],
                    "predict": " ".join(prediction),
                    # "final_layer": fsinal_layer[p],
                }
                out_data[data[start_id]["id"]] = data_point

                pbar.update(1)
                start_id += 1
        else:
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((i + start_id, src_tokens_i, hypos))

            # sort output to match input order
            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.common_eval.post_process)

                # Process top predictions
                for hypo_id, hypo in enumerate(hypos):
                    _hypo_tokens, hypo_str, _alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.common_eval.post_process,
                    )

                    detok_hypo_str = hypo_str
                    utterance = detok_hypo_str
                    prediction = utterance.removeprefix(src_str).strip()
                    # assert utterance != prediction  # important when the generated sequence (max-len-b) is shorter than src

                    data_point = {
                        "file_name": data[id]["file_name"],
                        "src": data[id]["src"],
                        "label": data[id]["label"],
                        "predict": prediction,
                    }

                    out_data[data[id]["id"]] = data_point
                pbar.update(1)
            start_id += len(results)

    # === write result === #
    with open(output_file, "w") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)


def cli_main(parser, input_args):
    args = options.parse_args_and_arch(parser, input_args)

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    main(args)


if __name__ == "__main__":
    cli_main()
