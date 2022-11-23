import os
import time
import yaml
from utils import parameter
from train import train
from U1652_test_and_evaluate import eval_and_test


def Auto_tune(drop_rate, learning_rate):
    # for model in model_list:
    #     parameter("model", model)
        for dr in drop_rate:
            parameter("drop_rate", dr)
            for lr in learning_rate:
                parameter("lr", lr)
                # for wd in weight_decay:
                #     parameter("weight_decay", wd)
                with open("settings.yaml", "r", encoding="utf-8") as f:
                    setting_dict = yaml.load(f, Loader=yaml.FullLoader)
                    print(setting_dict)
                    f.close()
                train()
                try:
                    eval_and_test(384)
                except:
                    print("error")
                    continue


# height_list = [150, 200, 250, 300]
learning_rate = [0.008, 0.009, 0.01]
drop_rate = [0.2, 0.25]

# model_list = ["LPN"]
Auto_tune(drop_rate, learning_rate)
