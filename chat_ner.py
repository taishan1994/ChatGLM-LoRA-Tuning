import torch
import json
from pprint import pprint
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

data_name = "msra"

train_args_path = "./checkpoint/{}/train_trainer/adapter_model/train_args.json".format(data_name)
with open(train_args_path, "r") as fp:
    args = json.load(fp)


config = AutoConfig.from_pretrained(args["model_dir"], trust_remote_code=True)
pprint(config)
tokenizer = AutoTokenizer.from_pretrained(args["model_dir"],  trust_remote_code=True)

model = AutoModel.from_pretrained(args["model_dir"],  trust_remote_code=True).half().cuda()
model = model.eval()
model = PeftModel.from_pretrained(model, args["save_dir"], torch_dtype=torch.float32, trust_remote_code=True)
model.half().cuda()
model.eval()

while True:
    inp = input("用户 >>> ")
    response, history = model.chat(tokenizer, inp, history=[])
    print("ChatNER >>> ", response))
    print("="*100)