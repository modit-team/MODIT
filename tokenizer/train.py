import sentencepiece as spm
import sys

import os



# /proj/arise/arise/saikat/CodeChangeDataSetTufano/small/test
# data.child_full_code data.commit_msg  data.parent_code  data.parent_seqr

data = sys.argv[1]
print(data)

if data == 'tufano':
    FILES = []
    ds = ['medium', 'small']
    part = ['train', 'eval', 'test']
    base_dir = '/proj/arise/arise/saikat/CodeChangeDataSetTufano/'
    fs = ['data.child_full_code', 'data.commit_msg', 'data.parent_seqr']
    for d in ds:
        for p in part:
            for f in fs:
                FILES.append(os.path.join(base_dir, d, p, f))
                pass
            pass
        pass
    model_prefix = 'tufano'
    pass
else:
    FILES = ['data.txt']
    model_prefix = 'vocab'


for vs in [10000, 20000, 30000, 50000]:
    if not os.path.exists('%s.%d' % (model_prefix, vs)):
        os.mkdir('%s.%d' % (model_prefix, vs))
    spm.SentencePieceTrainer.train(
        '--input=%s --vocab_size=%d --model_prefix=%s.%d/spm --character_coverage=1.0 --model_type=bpe --max_sentence_length=10000' % (','.join(FILES), vs, model_prefix, vs)
    )
