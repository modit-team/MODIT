import json
import os
import numpy as np
from dateutil.parser import parse
import sys


class JsonProgress(object):
    def __init__(self):
        self.count = 0

    def __call__(self, obj):
        self.count += 1
        sys.stdout.write("\r%8d" % self.count)
        return obj


def open_all_files(dir_path, file_names):
    files = []
    for fn in file_names:
        files.append(open(os.path.join(dir_path, fn)))
    return files
    pass


def read_all_data(dir_path, file_names):
    taken_data = set()
    data = []
    files = open_all_files(dir_path, file_names)
    for idx, (child_code, child_full_code, child_full_tree, child_tree, commit_msg, commit_time,
              file_name, parent_code, parent_seqr, parent_tree) in enumerate(zip(*files)):
        try:
            if idx % 10000 == 0:
                print('Read %d examples so far' % idx)
            key = parent_code + " " + child_code
            if key in taken_data:
                continue
            parts = file_name.strip().strip('/').split('/')
            project = parts[6]
            commit_id = parts[7]
            file_path = parts[-1].replace('_', '/')
            data.append({
                'project': project,
                'commit.id': commit_id,
                'commit.time': commit_time.strip(),
                'commit.msg': commit_msg.strip(),
                'file_path': file_path,
                'parent.code': parent_code.strip(),
                'parent.tree': json.loads(parent_tree.strip()),
                'child.full.code': child_full_code.strip(),
                'child.full.tree': json.loads(child_full_tree.strip()),
                'child.code': child_code.strip(),
                'child.tree': json.loads(child_tree.strip()),
                'parent.seqr': parent_seqr.strip()
            })
            taken_data.add(key)
        except:
            pass
        pass
    print('Total Examples: %d' % len(data))
    return data
    pass


def save_data(dir_name, train_data, valid_data, test_data):
    print('Saving to %s' % dir_name)
    print('Train:\t%d' % len(train_data))
    print('Valid:\t%d' % len(valid_data))
    print('Test:\t%d' % len(test_data))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, 'train.json'), 'w') as t:
        json.dump(train_data, t)
        t.close()
        pass
    with open(os.path.join(dir_name, 'valid.json'), 'w') as t:
        json.dump(valid_data, t)
        t.close()
        pass
    with open(os.path.join(dir_name, 'test.json'), 'w') as t:
        json.dump(test_data, t)
        t.close()
    pass


def sort_and_split_based_on_time(data):
    times = [(d, parse(d['commit.time'])) for d in data]
    sorted_tuples = sorted(times, key=lambda x:x[1])
    total = len(sorted_tuples)
    train_number = int(0.7*total)
    train = sorted_tuples[:train_number]
    train_last_time = train[1]
    while sorted_tuples[train_number][1] == train_last_time:
        train.append(sorted_tuples[train_number])
        train_number += 1
        pass
    valid_number = int(0.8*total) - 1
    valid_last_time = sorted_tuples[valid_number - 1][1]
    while sorted_tuples[valid_number][1] == valid_last_time and valid_number > train_number:
        valid_number -= 1
        pass
    if train_number < valid_number:
        valid = sorted_tuples[train_number:valid_number]
        test = sorted_tuples[valid_number:]
        pass
    else:
        valid = []
        test = []
    return [d[0] for d in train], [d[0] for d in valid], [d[0] for d in test]
    pass


def save_timeline_split(full_data, dir_name):
    project_to_data = {}
    train_data, valid_data, test_data = [], [], []
    for d in full_data:
        if d['project'] not in project_to_data:
            project_to_data[d['project']] = []
        project_to_data[d['project']].append(d)
    for p in project_to_data:
        train, valid, test = sort_and_split_based_on_time(project_to_data[p])
        print('Project: %s\tTrain: %d\tValid: %d\tTest: %d' % (p, len(train), len(valid), len(test)))
        train_data.extend(train)
        valid_data.extend(valid)
        test_data.extend(test)
        pass
    save_data(dir_name, train_data, valid_data, test_data)


def save_random_split(full_data, dir_name):
    np.random.shuffle(full_data)
    total = len(full_data)
    train_data = full_data[:int(0.7*total)]
    valid_data = full_data[int(0.7 * total):int(0.8 * total)]
    test_data = full_data[int(0.8 * total):]
    save_data(dir_name, train_data, valid_data, test_data)
    pass


if __name__ == '__main__':
    file_names = ['child.code',  'child.full.code',  'child.full.tree',  'child.tree',
                  'commit.msg',  'commit.time',  'files.txt',  'parent.code',
                  'parent.seqr',  'parent.tree']
    full_data = read_all_data('/proj/arise/arise/saikat/defj_train_data/data/all/full_data',
                              file_names)
    #print('Saving to full_data.json')
    #output = open('full_data.json', 'w')
    #json.dump(full_data, output)
    #output.close()

    split_count = 5
    #for sp in range(split_count):
    #    save_random_split(full_data, dir_name='random_splits/split-'+str(sp+5))
    #    pass
    save_timeline_split(full_data, dir_name='time_split')
    pass
