import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import load_data, NerCollate
from transformers import AutoModel, AutoTokenizer
from config_utils import ConfigParser
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

data_name = "msra"

train_args_path = "./checkpoint/{}/train_trainer/adapter_model/train_args.json".format(data_name)
with open(train_args_path, "r") as fp:
    args = json.load(fp)

config_parser = ConfigParser(args)
args = config_parser.parse_main()

model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
model.eval()
model = PeftModel.from_pretrained(model, os.path.join(args.save_dir, "adapter_model"), torch_dtype=torch.float32, trust_remote_code=True)
model.half().cuda()
model.eval()

# 找到labels中预测开始的部分
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

texts = [
    {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
     "query": "文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。", "answer": "郑振铎_人名\n阿英_人名\n国民党_机构名"},
    {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
     "query": "文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。", "answer": "北京市_地名"},
    {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
     "query": "文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。", "answer": "姜德明_人名"},
]

test_data = texts
ner_collate = NerCollate(args, tokenizer)
test_dataloader = DataLoader(test_data,
                             batch_size=args.train_batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=ner_collate.collate_fn)

with torch.no_grad():
    all_preds = []
    all_trues = []
    for step, batch in enumerate(test_dataloader):
        for k,v in batch.items():
            batch[k] = v.cuda()
        labels = batch["labels"].detach().cpu().numpy()

        output = model(**batch)
        logits = output.logits
        preds = torch.argmax(logits, -1).detach().cpu().numpy()
        preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
        for pre, lab in zip(preds.tolist(), labels.tolist()):
            d1 = tokenizer.convert_ids_to_tokens(lab)
            d2 = tokenizer.convert_ids_to_tokens(pre)
            print(d1)
            print(d2)
            for dd1, dd2 in zip(d1, d2):
                print(dd1, dd2)
            print("="*100)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, decoder_end_token_id=eos_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, decoder_end_token_id=eos_token_id)
        all_preds.extend(decoded_preds)
        all_trues.extend(decoded_labels)

for text, pred, true in zip(texts, all_preds, all_trues):
    text = json.dumps(text["instruct"] + text["query"], ensure_ascii=False)
    print("文本 >>> ", text)
    print("预测 >>> ", json.dumps(pred, ensure_ascii=False))
    print("真实 >>> ", json.dumps(true, ensure_ascii=False))

