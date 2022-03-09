from pytorch_lightning.metrics.functional import auc
from pytorch_lightning.metrics.functional.classification import precision_recall_curve
import torch
import argparse
import os
import matplotlib.pyplot as plt


def compute_auc(precisions, recalls,
				filter_threshold: float = 0.5):
	xs, ys = [], []
	for p, r in zip(precisions, recalls):
		if p >= filter_threshold:
			xs.append(r)
			ys.append(p)

	return auc(
		torch.cat([x.unsqueeze(0) for x in xs], 0),
		torch.cat([y.unsqueeze(0) for y in ys], 0)
	)


def find_best_F1(precisions, recalls, thresholds):
	def calc_f1(p, r):
		return 2 * (p * r) /(p + r)

	best_f1 = 0.0
	best_thres = None
	best_prec = None
	best_rec = None

	for p, r, t in zip(precisions, recalls, thresholds):
		cur_f1 = calc_f1(p, r)
		if cur_f1 > best_f1:
			best_f1 = cur_f1
			best_thres = t
			best_prec = p
			best_rec = r
	if best_thres is None:
		print(f"Attention! No f1 score is found to be non-zero!")
	return best_f1, best_prec, best_rec, best_thres


def get_SS_style_result(score_fn, filter_thres=0.5, pr_rec_ofn=None):
	gold = []
	pred = []
	with open(score_fn, 'r', encoding='utf8') as fp:
		for line in fp:
			g, p = line.split(' ')
			g = int(float(g))
			p = float(p)
			gold.append(g)
			pred.append(p)

	assert len(gold) == len(pred)
	gold = torch.Tensor(gold)
	pred = torch.Tensor(pred)

	prec, rec, thres = precision_recall_curve(pred, gold)
	if pr_rec_ofn is not None:
		with open(pr_rec_ofn, 'w', encoding='utf8') as ofp:
			for p, r, t in zip(prec, rec, thres):
				ofp.write(f'{p}\t{r}\t{t}\n')

	auc_var = compute_auc(prec, rec, filter_thres)
	auc_half = compute_auc(prec, rec, 0.5)
	rel_prec = torch.tensor([max(p - filter_thres, 0) for p in prec], dtype=torch.float)
	rel_rec = torch.tensor([r for r in rec], dtype=torch.float)
	area_under_pr_rec_curve_relative = compute_auc(
		rel_prec, rel_rec,
		filter_threshold=0.0
	)
	area_under_pr_rec_curve_relative /= (1 - filter_thres)

	best_f1, best_prec, best_rec, best_thres = find_best_F1(prec, rec, thres)

	metrics = {
		'best F1': best_f1,
		'Precision': best_prec, 'Recall': best_rec,
		'AUC_BSLN': auc_var,
		'AUCHALF': auc_half,
		'AUCNORM': area_under_pr_rec_curve_relative,
		'theta': best_thres
	}

	print(f"Metrics: ")
	print(metrics)

	plt.plot(rec, prec, label='')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.title("Precision Recall Curves")
	plt.legend()
	plt.draw()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', type=str, default='pr_rec_orig_test_exhaust_22')
	parser.add_argument('--scr_fn', type=str, default='global_scores_orig_test_apooling_binc_Y.txt')
	parser.add_argument('--subset', type=str, default='lh')

	args = parser.parse_args()

	if args.subset == 'lh':
		filter_thres = 0.21910
	elif args.subset == 'lhdir':
		filter_thres = 0.5
	elif args.subset == 'sh':
		filter_thres = 0.33255
	else:
		raise AssertionError

	get_SS_style_result(os.path.join(args.root, args.scr_fn), filter_thres)
