import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='false_pos_diff_max_0.50.txt', type=str)
parser.add_argument('--target', default='false_pos_diff_avg_1.00.txt', type=str)

args = parser.parse_args()

source_res = {'SS': [], 'ST': [], 'EE': [], 'RE': [], 'TE': []}

with open(args.source, 'r', encoding='utf8') as fp:
	newent_flag = True
	cur_tuple = []
	cur_label = None
	for line in fp:
		line = line.strip('\n')
		if newent_flag is True:
			if len(line) == 0:
				continue
			assert line == '-----------------------------------------------'
			newent_flag = False
			continue
		if line in ['SS', 'ST', 'EE', 'RE', 'TE']:
			assert cur_label is None
			cur_label = line
		elif len(line) == 0:
			source_res[cur_label].append(cur_tuple)
			newent_flag = True
			cur_tuple = []
			cur_label = None
		else:
			cur_tuple.append(line)

	if cur_label is not None:
		source_res[cur_label].append(cur_tuple)
		print(f"last entry: {cur_label}: {cur_tuple}")

with open(args.target, 'r', encoding='utf8') as fp:
	target_lines = fp.readlines()

with open(args.target, 'w', encoding='utf8') as fp:
	newent_flag = True
	cur_tuple = []
	cur_label = None
	for line in target_lines:
		line = line.rstrip('\n')
		assert line not in ['SS', 'ST', 'EE', 'RE', 'TE']
		if newent_flag is True:
			if len(line) == 0:
				fp.write(line+'\n')
				continue
			assert line == '-----------------------------------------------'
			newent_flag = False
			fp.write(line+'\n')
			continue
		if len(line) == 0:
			for key in source_res:
				if cur_label is not None:
					break
				for tup in source_res[key]:
					assert len(tup) == len(cur_tuple)
					matched = True
					for s, t in zip(tup, cur_tuple):
						if s != t:
							matched = False
							break
					if matched:
						cur_label = key
						break
			for l in cur_tuple:
				fp.write(l+'\n')
			if cur_label is not None:
				fp.write(cur_label+'\n')
			else:
				fp.write('\n')
			fp.write('\n')
			cur_tuple = []
			cur_label = None
			newent_flag = True
		else:
			cur_tuple.append(line)

