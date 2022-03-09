from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print
import sys
sys.path.append("..")
sys.path.append("../../DrQA/")
sys.path.append("/Users/teddy/PycharmProjects/DrQA/")
from graph import graph
from drqa.retriever.tfidf_doc_ranker import TfidfDocRanker
from qaeval_chinese_general_functions import load_data_entries, type_matched, reconstruct_sent_from_rel, \
	calc_per_entry_score_bert, find_entailment_matches_from_graph
import evaluation.util_chinese
from sklearn.metrics import precision_recall_curve

import os
import time
import json
import torch
import transformers


def qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
								   entry_uscores=None, gr=None, loaded_data_refs_by_partition=None,
								   loaded_ref_triples_by_partition=None, suppress=False):
	if args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
		with torch.no_grad():
			bert_tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_dir)
			bert_model = transformers.BertModel.from_pretrained(args.bert_dir)
			bert_model.eval()
			args.device = torch.device(args.device_name) if torch.cuda.is_available() else torch.device('cpu')
			bert_model = bert_model.to(args.device)
		if args.eval_method in ['bert1A']:
			ranker = TfidfDocRanker(tfidf_path=args.tfidf_path, articleIds_by_partition_path=args.articleIds_dict_path,
									strict=False)
		else:
			ranker = None
	elif args.eval_method in ['eg']:
		bert_tokenizer = None
		bert_model = None
		ranker = None
		assert gr is not None
	else:
		raise AssertionError

	dur_loadtriples = 0.0
	this_total_num_matches = 0
	sum_data_refs = 0.0
	sum_data = 0.0
	sum_typed_ents = 0.0
	sum_typematches = 0.0

	if loaded_data_refs_by_partition is None:
		load_data_refs_flag = False
		store_data_refs_flag = False
	elif len(loaded_data_refs_by_partition) == 0:
		assert isinstance(loaded_data_refs_by_partition, dict)
		load_data_refs_flag = False
		store_data_refs_flag = True
	else:
		assert len(loaded_data_refs_by_partition) == len(date_slices) or args.debug
		load_data_refs_flag = True
		store_data_refs_flag = False

	if loaded_ref_triples_by_partition is None:
		load_triples_flag = False
		store_triples_flag = False
	elif len(loaded_ref_triples_by_partition) == 0:
		assert isinstance(loaded_ref_triples_by_partition, dict)
		load_triples_flag = False
		store_triples_flag = True
	else:
		assert len(loaded_ref_triples_by_partition) == len(date_slices) or args.debug
		load_triples_flag = True
		store_triples_flag = False

	for partition_key in date_slices:

		if args.debug and partition_key != '07-26_07-28':
			print(f"Processing only partition ``07-26_07-28'', skipping current partition!")
			continue

		if not suppress:
			print(f"Processing partition {partition_key}! Loading time so far: {dur_loadtriples} seconds")
		partition_triple_path = os.path.join(args.sliced_triples_dir,
											 args.sliced_triples_base_fn % (args.slicing_method, partition_key))
		partition_triples_in_sents = []

		st_loadtriples = time.time()
		if load_triples_flag:
			partition_triples_in_sents = loaded_ref_triples_by_partition[partition_key]
		else:
			with open(partition_triple_path, 'r', encoding='utf8') as fp:
				for line in fp:
					item = json.loads(line)
					partition_triples_in_sents.append(item)
		if store_triples_flag:
			loaded_ref_triples_by_partition[partition_key] = partition_triples_in_sents
		else:
			pass
		et_loadtriples = time.time()
		dur_loadtriples += (et_loadtriples - st_loadtriples)

		cur_partition_docids_to_in_partition_sidxes = {}  # This dict is only populated and used in bert1 setting!

		# build up the current-partition-dataset
		cur_partition_data_entries = []
		cur_partition_data_refs = []  # this does not change across different graphs, and can be computed once and loaded each time afterwards!
		cur_partition_global_dids = []
		cur_partition_typematched_flags = []
		for iid, item in enumerate(data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(item)
			tm_flag = None
			if gr is not None and not type_matched(gr.types, tsubj, tobj):
				tm_flag = False
			else:
				tm_flag = True
				this_total_num_matches += 1

			if item['partition_key'] == partition_key:
				cur_partition_data_entries.append(item)
				cur_partition_data_refs.append([])
				cur_partition_global_dids.append(iid)
				cur_partition_typematched_flags.append(tm_flag)

		# build up entity-pair dict
		ep_to_cur_partition_dids = {}
		for cid, ent in enumerate(cur_partition_data_entries):
			upred, subj, obj, tsubj, tobj = parse_rel(ent)
			ep_fwd = '::'.join([subj, obj])
			ep_rev = '::'.join([obj, subj])
			if ep_fwd not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_fwd] = []
			ep_to_cur_partition_dids[ep_fwd].append((cid, True, upred))

			if ep_rev not in ep_to_cur_partition_dids:
				ep_to_cur_partition_dids[ep_rev] = []
			ep_to_cur_partition_dids[ep_rev].append((cid, False, upred))

		if args.eval_method not in ['bert1A']:
			# sort out the related sent and rel ids for each entity pair
			# if ``delete-same-rel-sents'', delete the sentences with the exact same relations.
			# (or maybe put that into a filter for positives, the occurrence of entity pairs do not count the ones with the same predicate.)
			if load_data_refs_flag is True:
				assert len(loaded_data_refs_by_partition[partition_key]) == len(cur_partition_data_refs)
				cur_partition_data_refs = loaded_data_refs_by_partition[partition_key]
			else:
				for sidx, sent_item in enumerate(partition_triples_in_sents):
					if args.debug and sidx > 100000:
						break
					for ridx, r in enumerate(sent_item['rels']):
						rupred, rsubj, robj, rtsubj, rtobj = parse_rel(r)
						r_ep = '::'.join([rsubj,
										  robj])  # reference entity pair may be in the same order or reversed order w.r.t. the queried entity pair.
						if r_ep in ep_to_cur_partition_dids:
							for (cur_partition_did, aligned, query_upred) in ep_to_cur_partition_dids[r_ep]:
								assert isinstance(aligned, bool)

								# skip that sentence where the query is found! Also skip those relations that are the same as the query relation.
								# TODO: but leave those sentences that have the exact match relations to the query relation be!
								# TODO: if there are other relations in those sentences, and they are extracted, then these sentences
								# TODO: would still be used as part of context!
								if sidx != cur_partition_data_entries[cur_partition_did]['in_partition_sidx']:
									if (not args.keep_same_rel_sents) and query_upred == rupred:
										if args.debug:
											# print(f"Same predicate!")
											pass
										pass
									else:
										cur_partition_data_refs[cur_partition_did].append((sidx, ridx, aligned))
								else:
									if args.debug:
										# print(f"Same sentence: ref rel: {r}; query rel: {cur_partition_data_entries[cur_partition_did]['r']}")
										pass
			if store_data_refs_flag:
				assert loaded_data_refs_by_partition is not None
				loaded_data_refs_by_partition[partition_key] = cur_partition_data_refs
			else:
				pass
		else:  # if args.eval_method in ['bert1A']
			for sidx, sent_item in enumerate(partition_triples_in_sents):
				# Can't do the early stopping below! Will cause sentences to be unmatched for Bert1 method!
				# if args.debug and sidx > 100000:
				# 	break
				sent_docid = str(sent_item['articleId'])
				if sent_docid not in cur_partition_docids_to_in_partition_sidxes:
					cur_partition_docids_to_in_partition_sidxes[sent_docid] = []
				assert sidx not in cur_partition_docids_to_in_partition_sidxes[sent_docid]
				cur_partition_docids_to_in_partition_sidxes[sent_docid].append(sidx)

		for cid, reflst in enumerate(cur_partition_data_refs):
			sum_data += 1
			sum_data_refs += len(reflst)

		st_calcscore = time.time()
		# calculate the confidence value for each entry
		for cid, ent in enumerate(cur_partition_data_entries):
			if cid % 2000 == 1:
				ct_calcscore = time.time()
				dur_calcscore = ct_calcscore - st_calcscore
				print(f"calculating score for data entry {cid} / {len(cur_partition_data_entries)} for current partition;")
				duration_format_print(dur_calcscore, '')

			cur_score = None
			if args.eval_method == 'bert1A':
				ref_sents = []
				query_sent = reconstruct_sent_from_rel(ent, args.max_spansize)
				ref_docids, ref_tfidf_scrs = ranker.closest_docs(query_sent, partition_key=ent['partition_key'], k=args.num_refs_bert1)
				assert len(ref_docids) <= args.num_refs_bert1
				for rdid in ref_docids:
					# print(rdid)
					rsidxes = cur_partition_docids_to_in_partition_sidxes[rdid]
					for rsidx in rsidxes:
						if rsidx != ent['in_partition_sidx']:
							ref_sents.append(partition_triples_in_sents[rsidx]['s'])
				cur_score = calc_per_entry_score_bert(ent, ref_rels=None, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  max_context_size=args.max_context_size, debug=args.debug)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
			elif args.eval_method in ['bert2A', 'bert3A']:
				ref_rels = []
				ref_sents = []
				# for Bert methods, ``aligned'' var is not used: whether or not the entity pairs are aligned is unimportant for Bert.
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					ref_sents.append(partition_triples_in_sents[sid]['s'])
					ref_rels.append(partition_triples_in_sents[sid]['rels'][rid])
				cur_score = calc_per_entry_score_bert(ent, ref_rels=ref_rels, ref_sents=ref_sents,
													  method=args.eval_method,
													  max_spansize=args.max_spansize, bert_model=bert_model,
													  bert_tokenizer=bert_tokenizer, bert_device=args.device,
													  max_context_size=args.max_context_size, debug=args.debug)
				assert cur_partition_typematched_flags[cid] is True
				cur_uscore = None
			elif args.eval_method in ['eg']:
				ref_rels = []
				for (sid, rid, aligned) in cur_partition_data_refs[cid]:
					this_rel = partition_triples_in_sents[sid]['rels'][rid]
					this_rel['aligned'] = aligned
					ref_rels.append(this_rel)
				cur_score, cur_uscore, cur_num_true_entailments = find_entailment_matches_from_graph(gr, ent, ref_rels,
																		   cur_partition_typematched_flags[cid],
																		   feat_idx=args.eg_feat_idx, debug=args.debug)
				if cur_num_true_entailments is not None:
					sum_typed_ents += cur_num_true_entailments
					sum_typematches += 1
			else:
				raise AssertionError

			# The condition below means, either the current eval_method is some Bert method, or the current entry matches
			# the type of the current graph
			if cur_partition_typematched_flags[cid] is True:
				assert entry_tscores[cur_partition_global_dids[cid]] is None
				assert entry_processed_flags[cur_partition_global_dids[cid]] is False
				if cur_score is None:
					cur_score = 0.0
				entry_tscores[cur_partition_global_dids[cid]] = cur_score
				entry_processed_flags[cur_partition_global_dids[cid]] = True
			else:
				assert cur_score is None

			# The condition below means, the eval_method is EG, and some non-zero entailment score has been found between
			# the query rel and some reference rel in this type pair. (the uscore means this is ignoring type, we'll average them later)
			# TODO: double check whether backupAvg indeed means backup to the average of all type pairs where some non-zero
			# TODO: entailment score has been found.
			# ⬆️ it is indeed: the predPairFeats were the sum of all entailment scores where some entailment score other
			# than ``None'' was returned; later on it is divided by the value in predPairSumCoefs, which is the number of
			# such entailment scores as described above.
			# TODO: NOTE! The ``other than None'' includes that cases where all-zeros are returned. These cases mean that
			# TODO: both predicates are found in the graph, but no edges connect between them. The meaning of this is that,
			# TODO: this sub-graph does not think there exists an edge between this pair of predicates, that opinion matters,
			# TODO: so this zero-score should be counted in the denominator, and should not be ignored.
			if cur_uscore is not None and ((not args.ignore_0_for_Avg) or cur_uscore > 0.0000000001):  # a small number, not zero for numerical stability
				entry_uscores[cur_partition_global_dids[cid]].append(cur_uscore)

	if sum_data > 0:
		avg_refs_per_entry = sum_data_refs / sum_data
		print(f"Average number of references per entry: {avg_refs_per_entry}")
	else:
		print(f"Anomaly! sum_data not larger than zero! sum_data: {sum_data}; sum_data_refs: {sum_data_refs}.")
	if sum_typematches > 0:
		avg_typed_ents = sum_typed_ents / sum_typematches
		print(f"Average number of typed entailment edges utilized: {avg_typed_ents}")
	else:
		print("No type match found! avg_typed_ents equals to 0.")

	return dur_loadtriples, this_total_num_matches


def qa_eval_boolean_main(args, date_slices):
	data_entries = load_data_entries(args.fpath, posi_only=False)

	entry_processed_flags = [False for x in range(len(data_entries))]
	entry_tscores = [None for x in range(len(data_entries))]
	entry_uscores = [[] for x in range(len(data_entries))]

	all_tps = []  # all_type_pairs

	total_dur_loadtriples = 0.0

	# There are two loops: one for all entailment sub-graphs, the other for all partitions.
	# Both are too large to store all in memory at once, and entGraphs take longer to load.
	# So in the outer loop, iterate over all type-pairs; for each type pair, retrieve results from the corresponding subgraphs

	if args.eval_method == 'eg':
		if args and args.eg_feat_idx is not None:
			graph.Graph.featIdx = args.eg_feat_idx

		files = os.listdir(args.eg_dir)
		files.sort()
		num_type_pairs_processed = 0
		num_type_pairs_processed_reported_flag = False

		loaded_data_refs_by_partition = None if args.no_ref_cache else {}
		loaded_ref_triples_by_partition = None if args.no_triple_cache else {}

		for f in files:
			if num_type_pairs_processed % 50 == 1 and not num_type_pairs_processed_reported_flag:
				print(f"num processed type pairs: {num_type_pairs_processed}")
				num_type_pairs_processed_reported_flag = True
			if not f.endswith(args.eg_suff):
				continue
			gpath = os.path.join(args.eg_dir, f)
			if os.path.getsize(gpath) < args.min_graphsize:
				continue
			gr = graph.Graph(gpath=gpath, args=args)
			gr.set_Ws()
			all_tps.append(gr.types)

			cur_dur_loadtriples, this_num_matches = qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
										   entry_uscores=entry_uscores, gr=gr, loaded_data_refs_by_partition=loaded_data_refs_by_partition,
										   loaded_ref_triples_by_partition=loaded_ref_triples_by_partition, suppress=True)
			total_dur_loadtriples += cur_dur_loadtriples
			num_type_pairs_processed += 1
			num_type_pairs_processed_reported_flag = False
			this_percent_matches = '%.2f' % (100 * this_num_matches / len(data_entries))
			print(f"Finished processing for graph of types: {gr.types[0]}#{gr.types[1]}; num of entries matched: {this_num_matches} -> {this_percent_matches} percents of all entries.")
	elif args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
		total_dur_loadtriples, _ = qa_eval_boolean_all_partitions(args, date_slices, data_entries, entry_processed_flags, entry_tscores,
									   entry_uscores=None, gr=None)
	else:
		raise AssertionError

	duration_format_print(total_dur_loadtriples, f"Total duration for loading triples")

	assert len(entry_processed_flags) == len(entry_tscores)
	unmatched_types = set()

	for eidx, (fl, sc) in enumerate(zip(entry_processed_flags, entry_tscores)):
		if args.eval_method in ['eg']:
			entry_rel = data_entries[eidx]
			upred, subj, obj, tsubj, tobj = parse_rel(entry_rel)
			matched_flag = False
			for tp in all_tps:
				if type_matched(tp, tsubj, tobj):
					matched_flag = True
			if matched_flag is False:
				if ('::'.join([tsubj, tobj]) not in unmatched_types) and (
						'::'.join([tobj, tsubj]) not in unmatched_types):
					unmatched_types.add('::'.join([tsubj, tobj]))
				continue

		assert fl is True
		assert sc is not None

	if args.eval_method in ['eg']:
		print('unmatched types: ')
		print(unmatched_types)

	entry_avg_uscores = []
	for eidx, cur_uscores in enumerate(entry_uscores):
		avg_uscr = sum(cur_uscores) / float(len(cur_uscores)) if len(cur_uscores) > 0 else 0.0
		entry_avg_uscores.append(avg_uscr)
	assert len(entry_tscores) == len(entry_avg_uscores)

	# if args.store_skip_idxes:
	# 	with open(args.skip_idxes_fn, 'w', encoding='utf8') as fp:
	# 		skip_idxes = []
	# 		for eidx, flg in enumerate(entry_processed_flags):
	# 			if not flg:
	# 				skip_idxes.append(eidx)
	# 		json.dump(skip_idxes, fp, ensure_ascii=False)
	# else:
	# 	with open(args.skip_idxes_fn, 'r', encoding='utf8') as fp:
	# 		skip_idxes = json.load(fp)
	# 	for si in skip_idxes:
	# 		assert si < len(data_entries)
	# 	for eidx, flg in enumerate(entry_processed_flags):
	# 		assert eidx in skip_idxes or flg

	# this ``skipping those data entries whose type-pairs unmatched by any sub-graph'' thing, it should not be
	# necessary with backupAvg, and should not be reasonable without backupAvg. It kind of biases the evaluation.
	final_scores = []  # this is typed score if not backupAvg, and back-up-ed score if backupAvg
	final_labels = []
	for eidx, (tscr, uscr, ent) in enumerate(zip(entry_tscores, entry_avg_uscores, data_entries)):

		if tscr is not None and tscr > 0:
			final_scores.append(tscr)
		elif args.backupAvg and uscr is not None and uscr > 0:
			final_scores.append(uscr)
		else:
			final_scores.append(0.)
		if bool(ent['label']) is True:
			final_labels.append(1)
		elif bool(ent['label']) is False:
			final_labels.append(0)
		else:
			raise AssertionError
	assert len(final_labels) == len(final_scores) and len(final_labels) == len(data_entries)

	prec, rec, thres = precision_recall_curve(final_labels, final_scores)
	assert len(prec) == len(rec) and len(prec) == len(thres) + 1
	auc_value = evaluation.util_chinese.get_auc(prec[1:], rec[1:])
	print(f"Area under curve: {auc_value};")

	if args.eval_method in ['bert1A', 'bert2A', 'bert3A']:
		method_ident_str = args.eval_method
	elif args.eval_method in ['eg']:
		method_ident_str = '_'.join([args.eval_method, os.path.split(args.eg_name)[-1], args.eg_suff])
	else:
		raise AssertionError
	with open(args.boolean_predictions_path % method_ident_str, 'w', encoding='utf8') as fp:
		for t, u, s, l in zip(entry_tscores, entry_avg_uscores, final_scores, final_labels):
			fp.write(f"{t}\t{u}\t{s}\t{l}\n")

	with open(args.pr_rec_path % method_ident_str, 'w', encoding='utf8') as fp:
		fp.write(f"auc: {auc_value}\n")
		for p, r, t in zip(prec[1:], rec[1:], thres):
			fp.write(f"{p}\t{r}\t{t}\n")

	print(f"Finished!")