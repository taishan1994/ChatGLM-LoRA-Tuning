import re
import ast
import sys
from pprint import pprint


class ConfigParser:
    def __init__(self, config):
        self.config = config
        assert isinstance(config, dict)
        args = sys.argv
        args = args[1:]
        print(args)
        self.args = args

    def judge_type(self, value):
        """利用正则判断参数的类型"""
        if value.isdigit():
            return int(value)
        elif re.match(r'^-?\d+\.?\d*$', value):
            return float(value)
        elif value.lower() in ["true", "false"]:
            return True if value == "true" else False
        else:
            try:
                st = ast.literal_eval(value)
                return st
            except Exception as e:
                return value

    def get_args(self):
        return_args = {}
        for arg in self.args:
            arg = arg.split("=")
            arg_name, arg_value = arg
            if "--" in arg_name:
                arg_name = arg_name.split("--")[1]
            elif "-" in arg_name:
                arg_name = arg_name.split("-")[1]
            return_args[arg_name] = self.judge_type(arg_value)
        return return_args

    # 定义一个函数，用于递归获取字典的键
    def get_dict_keys(self, config, prefix=""):
        result = {}
        for k, v in config.items():
            new_key = prefix + "_" + k if prefix else k
            if isinstance(v, dict):
                result.update(self.get_dict_keys(v, new_key))
            else:
                result[new_key] = v
        return result

    # 定义一个函数，用于将嵌套字典转换为类的属性
    def dict_to_obj(self, merge_config):
        # 如果d是字典类型，则创建一个空类
        if isinstance(merge_config, dict):
            obj = type("", (), {})()
            # 将字典的键转换为类的属性，并将字典的值递归地转换为类的属性
            for k, v in merge_config.items():
                setattr(obj, k, self.dict_to_obj(v))
            return obj
        # 如果d不是字典类型，则直接返回d
        else:
            return merge_config

    def set_args(self, args, cls):
        """遍历赋值"""
        for key, value in args.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise Exception(f"参数【{key}】不在配置中，请检查！")
        return cls

    def parse_main(self):
        # 获取命令行输入的参数
        cmd_args = self.get_args()
        # 合并字典的键，用_进行连接
        merge_config = self.get_dict_keys(self.config)
        # 将字典配置转换为类可调用的方式
        class_config = self.dict_to_obj(merge_config)
        # 合并命令行参数到类中
        cls = self.set_args(cmd_args, class_config)
        return cls


if __name__ == '__main__':
    config = {
        "data_name": "msra",
        "ouput_dir": "./checkpoint/",
        "model_name": "bert",
        "do_predict": True,
        "do_eval": True,
        "do_test": True,
        "max_seq_len": 512,
        "lr_steps": [80, 180],
        "optimizer": {
            "adam": {
                "leraning_rate": 5e-3,
            },
            "adamw": {
                "leraning_rate": 5e-3,
            }
        }
    }

    print("最初参数：")
    pprint(config)
    config_parser = ConfigParser(config)
    args = config_parser.parse_main()
    print("修改后参数：")
    print("="*100)
    pprint(vars(args))

    print("="*100)
    print(args.model_name)
    print(args.optimizer_adam_leraning_rate)