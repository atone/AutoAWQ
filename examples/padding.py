import argparse
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def pad_model(model_path, output_path):
    model_path = Path(model_path)
    output_path = Path(output_path)

    # must use AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # this size is FLM-2-52B only
    pad_size = 704

    sd = model.state_dict()

    for k in tqdm(sd, desc='Padding model'):
        v = sd[k]
        # interleaving the padded zeros
        if ('mlp.up_proj.weight' in k) or ('mlp.gate_proj.weight' in k):
            prev_v = F.pad(v.unsqueeze(1), (0, 0, 0, 1, 0, 0)).reshape(21824*2, -1)[:pad_size*2]
            new_v = torch.cat([prev_v, v[pad_size:]], dim=0)
            sd[k] = new_v
        elif 'mlp.down_proj.weight' in k:
            prev_v= F.pad(v.unsqueeze(2), (0, 1)).reshape(-1, 21824*2)[:, :pad_size*2]
            new_v = torch.cat([prev_v, v[:, pad_size:]], dim=1)
            sd[k] = new_v

    # this is a very large file; make sure your RAM is enough to load the model
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Saving model to {output_path}')
    torch.save(sd, output_path / 'pytorch_model.bin')
    shutil.copy(model_path / 'modeling_teleflm.py', output_path / 'modeling_teleflm.py')
    # change the intermediate size and save the config
    model.config.intermediate_size = 22528
    model.config.save_pretrained(output_path)
    # save the tokenizer
    tokenizer.chat_template =   "{%- for message in messages %}" \
                                    "{%- if message['role'] == 'user' %}" \
                                        "{{- '<_user>' + message['content']|trim + '<_bot>' }}" \
                                    "{%- elif message['role'] == 'system' %}" \
                                        "{{- '<_system>' + message['content']|trim }}" \
                                    "{%- elif message['role'] == 'assistant' %}" \
                                        "{{- message['content'] }}" \
                                    "{%- endif %}" \
                                "{%- endfor %}"
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    pad_model(args.model_path, args.output_path)


if __name__ == '__main__':
    main()
