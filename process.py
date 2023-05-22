import json

def process_msra(in_file, out_file, mode=""):
  with open(in_file, 'r', encoding="utf-8") as fp:
        data = fp.readlines()
  ents = set()
  has_entity = []
  no_entity = []
  labels = ["人名", "地名", "机构名"]
  i = 0
  for d in data:
      d = json.loads(d)
      text = d["text"]
      if not text:
          continue
      entities = d["entity_list"]
      j = 0
      tmp = {}

      tmp["instruct"] = "你现在是一个实体识别模型，你需要提取文本里面的{}，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。".format("、".join(labels))
      tmp["query"] = "文本：" + text
      tmp["answer"] = "没有"
      if len(entities) == 0:
        no_entity.append(tmp)
        continue
      e_tmp = []
      for entity in entities:
          dtype = entity["entity_type"]
          e = entity["entity"]
          if dtype == "PER":
            dtype = "人名"
          elif dtype == "ORG":
            dtype = "机构名"
          elif dtype == "LOC":
            dtype = "地名"
          if e + "_" + dtype not in e_tmp:
            e_tmp.append(e + "_" + dtype)
      tmp["answer"] = "\n".join(e_tmp)
      has_entity.append(tmp)

  if mode == "train":
    print("有实体的数据：", len(has_entity))
    print("没尸体的数据：", len(no_entity))
    # train_data = has_entity[:2000] + no_entity[:500]
    train_data = has_entity + no_entity
    print(train_data[0])
    with open(out_file, "w") as fp:
      fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in train_data]))

  if mode == "dev":
    dev_data = has_entity
    print(dev_data[0])
    with open(out_file, "w") as fp:
      fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in dev_data]))


if __name__ == "__main__":
  process_msra("data/msra/ori_data/msra_train.txt", "data/msra/instruct_data/train.txt", mode="train")
  process_msra("data/msra/ori_data/msra_1000.txt", "data/msra/instruct_data/dev.txt", mode="dev")