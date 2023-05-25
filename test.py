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
model = PeftModel.from_pretrained(model, os.path.join(args.save_dir, "adapter_model"), torch_dtype=torch.float32, trust_remote_code=True)
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
    all_preds = []
    all_trues = []
    for step, batch in enumerate(tqdm(test_dataloader, ncols=100)):
        for k,v in batch.items():
            batch[k] = v.cuda()
        labels = batch["labels"].detach().cpu().numpy()

        output = model(**batch)
        logits = output.logits
        preds = torch.argmax(logits, -1).detach().cpu().numpy()
        preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
        preds = preds[:, :-1]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        all_preds.extend(decoded_preds)
        all_trues.extend(decoded_labels)

print("预测：", all_preds[:20])
print("真实：", all_trues[:20])
