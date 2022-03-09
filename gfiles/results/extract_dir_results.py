import argparse
import os
import json
from get_SS_style_auc import get_SS_style_result

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='pr_rec_orig_test_exhaust_22')
parser.add_argument('--fn', type=str, default='global_scores_orig_test_apooling_binc_Y.txt')
parser.add_argument('--ref_fn', type=str, default='./test_dir_idxes.json')
args = parser.parse_args()


lines = []
with open(os.path.join(args.root, args.fn), 'r', encoding='utf8') as fp:
	for line in fp:
		if line.rstrip('\n') != '':
			lines.append(line)

assert len(lines) == 12921

with open(args.ref_fn, 'r', encoding='utf8') as fp:
	dir_idxes = json.load(fp)

dir_lines = []

for lidx, line in enumerate(lines):
	if lidx in dir_idxes:
		dir_lines.append(line)

assert len(dir_lines) == len(dir_idxes)

assert args.fn[-5:] == 'Y.txt'
ofn = args.fn[:-5] + 'dir_Y.txt'
pr_rec_ofn = args.fn[:-5] + 'dir.txt'

with open(os.path.join(args.root, ofn), 'w', encoding='utf8') as ofp:
	for line in dir_lines:
		ofp.write(line)

get_SS_style_result(os.path.join(args.root, ofn), filter_thres=0.5, pr_rec_ofn=os.path.join(args.root, pr_rec_ofn))
