import json
import sys
import os
import argparse


def parse(input_file, output_file):
    lines = [line.strip() for line in input_file.readlines()]
    example_id_to_hyps = {}
    for line in lines:
        if line.startswith('S'):
            parts = line.split('\t')
            idx = int(parts[0].strip()[2:])
            source = parts[1]
            if idx not in example_id_to_hyps.keys():
                example_id_to_hyps[idx] = {
                    'source': source,
                    'target': "",
                    'hyps': []
                }
            else:
                example_id_to_hyps[idx]['source'] = source
            pass
        elif line.startswith('T'):
            parts = line.split('\t')
            idx = int(parts[0].strip()[2:])
            target = parts[1]
            if idx not in example_id_to_hyps.keys():
                example_id_to_hyps[idx] = {
                    'source': "",
                    'target': target,
                    'hyps': []
                }
            else:
                example_id_to_hyps[idx]['target'] = target
            pass
        elif line.startswith('H'):
            parts = line.split('\t')
            idx = int(parts[0].strip()[2:])
            score = float(parts[1])
            hyp = parts[2].strip()
            if idx not in example_id_to_hyps.keys():
                example_id_to_hyps[idx] = {
                    'source': "",
                    'target': "",
                    'hyps': [{'hyp': hyp, 'score': score}]
                }
            else:
                example_id_to_hyps[idx]['hyps'].append({'hyp': hyp, 'score': score})
            pass
        pass
    hyp_count = 0
    for index in sorted(example_id_to_hyps.keys()):
        example = example_id_to_hyps[index]
        hyps = sorted(example['hyps'], key=lambda x:x['score'])
        for hyp in hyps:
            output_file.write(hyp['hyp'].strip() + '\n')
            output_file.flush()
            hyp_count += 1
            pass
        pass
    print('Total Examples : %d' % len(example_id_to_hyps.keys()), file=sys.stderr)
    print('Total Hypotheses : %d' % hyp_count, file=sys.stderr)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Generated Input File', required=True)
    parser.add_argument('--output',
                        help='Output file path. Do not provide if want to print in console',
                        default=None)
    args = parser.parse_args()
    input_file = open(args.input)
    output_file = args.output
    if output_file is None:
        output_file = sys.stdout
        pass
    else:
        output_file = open(output_file, 'w')
    parse(input_file, output_file)
    if output_file is not sys.stdout or output_file is not sys.stderr:
        output_file.close()
    pass