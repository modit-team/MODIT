import json
from tqdm import tqdm

data = json.load(open('full_data.json'))

out = open('tokenizer/data.txt', 'w')


for d in tqdm(data):
    out.write(d['parent.code'] + '\n')
    out.write(d['child.full.code'] + '\n')
    out.write(d['commit.msg'] + '\n')
    pass

out.close()
