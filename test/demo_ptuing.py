import os
import time
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

model_dir = "/home/ubuntu/workspace/models/GLM-6B"
p_tuning_dir = "/home/ubuntu/workspace/models/GLM-6B/ptuning/adgen-1e-4/checkpoint-7000/"

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)

prefix_state_dict = torch.load(os.path.join(p_tuning_dir, "pytorch_model.bin"))

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


model = model.half().cuda()
model.transformer.prefix_encoder.float()

model = model.eval()

print("model load time consume:", time.time()-t0)


#warmup
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

print("command:")
inputs = '今天吃点什么？'
print(inputs)

while True:
    command = inputs
    t1 = time.time()

    response, history = model.chat(tokenizer, command, history=[])
    print(response)
    ret_count = len(response)
    infer_time_ms = round(1000*(time.time()-t1), 2)
    print("infer consume:{}ms, and speed:{} ms/word".format(infer_time_ms, infer_time_ms/ret_count))


    print('-'*40)
    print("command:(input something or press Ctrl +c to exit...)")
    inputs = input()

