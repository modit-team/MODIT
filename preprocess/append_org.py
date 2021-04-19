import sys
import json 
import os

data_dir = "/proj/arise/arise/saikat/experiment_data/CodeChangeGeneration/data/" + sys.argv[1].strip();

for part in ['train', 'valid', 'test']:
    parent_file = open(os.path.join(data_dir, part, 'data.parent_code'))
    commit_file = open(os.path.join(data_dir, part, 'data.commit_msg'))
    output_file = open(os.path.join(data_dir, part, 'data.parent_commit'), 'w')
    for p, c in zip(parent_file, commit_file):
        p = p.strip()
        c = c.strip('\"').strip()
        if c.endswith("\""):
            c = c[:-1].strip()
        sep = " <SEP> "
        t = p + sep + " " + c
        output_file.write(t + '\n')
    parent_file.close()
    commit_file.close()
    output_file.close()


