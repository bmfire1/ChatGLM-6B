from transformers import AutoTokenizer, AutoModel

model_dir = "/home/ubuntu/workspace/models/GLM-6B"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)


inputs = '今天吃点什么？'
while True:
    command = inputs
    response, history = model.chat(tokenizer, command, history=[])
    print(response)
    print('-'*40)
    print("command:")
    inputs = input()

