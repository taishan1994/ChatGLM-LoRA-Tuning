import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import load_data, NerCollate
from transformers import AutoModel, AutoTokenizer
from config_utils import ConfigParser
from decode_utils import get_entities
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


data_name = "msra"

train_args_path = "./checkpoint/{}/train_trainer/adapter_model/train_args.json".format(data_name)
with open(train_args_path, "r") as fp:
    args = json.load(fp)

config_parser = ConfigParser(args)
args = config_parser.parse_main()

model = AutoModel.from_pretrained(args.model_dir,  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
model.eval()
model = PeftModel.from_pretrained(model, args.save_dir, torch_dtype=torch.float32, trust_remote_code=True)
model.half().cuda()
model.eval()

test_data = load_data(args.dev_path)
ner_collate = NerCollate(args, tokenizer)
test_dataloader = DataLoader(test_data,
                  batch_size=args.train_batch_size,
                  shuffle=True,
                  drop_last=False,
                  collate_fn=ner_collate.collate_fn)

# 找到labels中预测开始的部分
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

with torch.no_grad():
    preds = []
    trues = []
    for step, batch in enumerate(tqdm(test_dataloader, ncols=100)):
        for k,v in batch.items():
            batch[k] = v.cuda()
        labels = batch["labels"].detach().cpu().numpy().tolist()

        output = model(**batch)
        logits = output.logits
        batch_size = logits.size(0)
        for i in range(batch_size):
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
            labels = batch["labels"][i].detach().cpu().numpy().tolist()
            start = labels.index(bos_token_id)
            end = labels.index(eos_token_id)
            pred_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(logits, -1)[i][start+1:end-1])
            true_tokens = tokenizer.convert_ids_to_tokens(labels[start+2:end])
            # 这里get_entities可以换成自己任务
            pred_res = get_entities(pred_tokens)
            true_res = get_entities(true_tokens)
            preds.append(pred_res)
            trues.append(true_res)

print("预测：", preds[:20])
print("真实：", trues[:20])