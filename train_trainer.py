import os
import json
import torch
import deepspeed
import argparse

from shutil import copy
from pprint import pprint
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import RandomSampler, DataLoader
from dataset import load_data, NerCollate
from config_utils import ConfigParser
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc.
            but create checkpoint folder (needed for saving adapter) """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value

                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class PeftSavingCallback(TrainerCallback):
    """ Correctly save PEFT model and not full model """

    def _save(self, model, folder):
        peft_model_path = os.path.join(folder, "adapter_model")
        model.save_pretrained(peft_model_path)

    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
                ):
        # checkpoint_folder = os.path.join(
        #     args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        # )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """ Save final best model adapter """
        self._save(kwargs['model'], state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """ Save intermediate model adapters in case of interrupted training """
        # folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self._save(kwargs['model'], args.output_dir)


def main():
    args = {
        "data_name": "msra",
        "model_dir": "/root/autodl-tmp/chatglm-6b/",
        "lora_r": 8,
        "max_source_length": 128,
        "max_target_length": 32,
        "instruct_column": "instruct",
        "query_column": "query",
        "response_column": "answer",
        "train_path": "data/msra/instruct_data/train.txt",
        "dev_path": "data/msra/instruct_data/dev.txt",
        "ignore_pad_token_for_loss": True,
        "train_batch_size": 12,
        "gradient_accumulation_steps": 1,
        "save_dir": "./checkpoint/msra/train_trainer/adapter_model/",
        "num_train_epochs": 1,
        "local_rank": -1,
        "log_steps": 10,
        "save_steps": 400,
        "deepspeed_json_path": "deepspeed.json",
    }

    config_parser = ConfigParser(args)
    args = config_parser.parse_main()

    pprint(vars(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "train_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    with open(args.deepspeed_jaon_path, "r") as fp:
        deepspeed_json = json.load(fp)

    import sys
    sys.exit(0)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir,
                                                  trust_remote_code=True,
                                                  )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, config)
    model = model.cuda()

    print_trainable_parameters(model)

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_data = load_data(args.train_path)
    ner_collate = NerCollate(args, tokenizer)

    train_datset = ner_collate.collate_fn(train_data)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        learning_rate=deepspeed_json["optimizer"]["params"]["lr"],
        adam_beta1=deepspeed_json["optimizer"]["params"]["betas"][0],
        adam_beta2=deepspeed_json["optimizer"]["params"]["betas"][1],
        weight_decay=deepspeed_json["optimizer"]["params"]["weight_decay"],
        fp16=deepspeed_json["fp16"],
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        save_total_limit=1,
        deepspeed=args.deepspeed_json_path,  # 设置 DeepSpeed 配置文件的路径

    )

    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=ner_collate.collate_fn,
        callbacks=[PeftSavingCallback],
    )

    trainer.train()


if __name__ == "__main__":
    main()

