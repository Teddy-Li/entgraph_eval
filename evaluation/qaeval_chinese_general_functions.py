import json
import psutil
import os
import copy
import random
random.seed()
import numpy as np

from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print


def load_data_entries(fpath, posi_only=False):
	data_entries = []
	with open(fpath, 'r', encoding='utf8') as fp:
		for line in fp:
			item = json.loads(line)
			assert item['label'] is not None and isinstance(item['label'], bool)
			# if the posi_only flag is set to True, then don't load the negatives! (This is reserved for wh-question answering (objects))
			if item['label'] is not True and posi_only:
				continue
			data_entries.append(item)
	process = psutil.Process(os.getpid())
	print(f"Current memory usage in bytes: {process.memory_info().rss}")  # in bytes
	return data_entries


def type_matched(types_lst_pointer, tsubj, tobj):
	types_lst = copy.deepcopy(types_lst_pointer)
	assert len(types_lst) == 2
	if types_lst[0][-2:] == '_1':
		assert types_lst[1][-2:] == '_2'
		types_lst[0] = types_lst[0][:-2]
		types_lst[1] = types_lst[1][:-2]
	if types_lst[0][-2:] == '_2':
		assert types_lst[1][-2:] == '_1'
		types_lst[0] = types_lst[0][:-2]
		types_lst[1] = types_lst[1][:-2]

	if tsubj[-2:] == '_1':
		assert tobj[-2:] == '_2'
		tsubj = tsubj[:-2]
		tobj = tobj[:-2]
	if tsubj[-2:] == '_2':
		assert tobj[-2:] == '_1'
		tsubj = tsubj[:-2]
		tobj = tobj[:-2]

	if tsubj == types_lst[0] and tobj == types_lst[1]:
		return True
	elif tsubj == types_lst[1] and tobj == types_lst[0]:
		return True
	else:
		return False


def mask_entities_for_entries(args):
	data_entries = load_data_entries(args.fpath, posi_only=True)
	out_fp = open(args.wh_fpath, 'w', encoding='utf8')

	for ent in data_entries:

		upred, subj, obj, tsubj, tobj = parse_rel(ent)
		arg2mask = None
		if args.mask_only_objs:
			arg2mask = 'obj'
		else:
			if random.random() > 0.5:
				arg2mask = 'subj'
			else:
				arg2mask = 'obj'

		if arg2mask == 'subj':
			ent['answer'] = subj
			ent['answer_type'] = tsubj
			ent['index_arg'] = obj
			ent['index_position'] = 'obj'
			ent['index_type'] = tobj
		elif arg2mask == 'obj':
			ent['answer'] = obj
			ent['answer_type'] = tobj
			ent['index_arg'] = subj
			ent['index_position'] = 'subj'
			ent['index_type'] = tsubj
		else:
			raise AssertionError

		out_line = json.dumps(ent, ensure_ascii=False)
		out_fp.write(out_line+'\n')

	out_fp.close()


def find_all_matches_in_string(string, pattern):
	id_list = []
	offset = 0
	while True:
		cur_id = string.find(pattern, offset)
		if cur_id < 0:
			break
		id_list.append(cur_id)
		offset = cur_id + len(pattern)
	return id_list


# From a given sentence, fetch the most relevant span to a given rel; if the rel is not given (None), just truncate the
# sentence to its first max_spansize tokens.
def fetch_span_by_rel(sent, rel, max_spansize):
	if len(sent) <= max_spansize:
		return sent
	if rel is None:
		return sent[:max_spansize]

	upred, subj, obj, tsubj, tobj = parse_rel(rel)
	subj_ids = find_all_matches_in_string(sent, subj)
	obj_ids = find_all_matches_in_string(sent, obj)

	if len(subj_ids) == 0 or len(obj_ids) == 0:
		return sent[:max_spansize]

	selected_sid = None
	selected_oid = None
	min_dist = len(sent)

	for sid in subj_ids:
		for oid in obj_ids:
			if abs(sid - oid) < min_dist:
				selected_sid = sid
				selected_oid = oid
				min_dist = abs(sid - oid)
	mid = int(selected_sid + selected_oid) // 2
	if mid < max_spansize // 2:
		return sent[:max_spansize]
	if mid > len(sent) - (max_spansize // 2):
		assert len(sent) - max_spansize > 0
		return sent[len(sent) - max_spansize:]

	assert mid - (max_spansize // 2) >= 0 and mid + (max_spansize // 2) <= len(sent)
	return sent[(mid - (max_spansize // 2)):(mid + (max_spansize // 2))]


def reconstruct_sent_from_rel(rel, max_spansize, mask_answer_flag=False):
	upred, subj, obj, tsubj, tobj = parse_rel(rel)

	# TODO: maybe later make this more fine-grained, like '什么', '谁', '什么时候', '哪里', (we don't have `how' here!)
	if mask_answer_flag:  # if mask_answer_flag, then mask the masked-argument with `[MASK]'
		assert 'index_position' in rel
		if rel['index_position'] == 'subj':
			obj = '[MASK]'  # then obj should be masked;
		elif rel['index_position'] == 'obj':
			subj = '[MASK]'  # then subj should be masked;
		else:
			raise AssertionError

	assert upred[0] == '(' and upred[-1] == ')'
	upred_surface_form = upred[1:-1]
	upred_surface_form = upred_surface_form.split('.1,')
	assert len(upred_surface_form) == 2
	upred_surface_form = upred_surface_form[0]
	upred_xidxs = find_all_matches_in_string(upred_surface_form, '·X·')

	if len(upred_xidxs) == 0:
		upred_surface_form = upred_surface_form.replace('·', '')
		reconstructed_sent = subj + upred_surface_form + obj
	elif len(upred_xidxs) == 1:
		upred_surface_form = upred_surface_form.replace('·X·', obj)
		upred_surface_form = upred_surface_form.replace('·', '')
		reconstructed_sent = subj + upred_surface_form  # we have stuck the object back into the predicate!
	else:
		raise AssertionError
	if len(reconstructed_sent) > max_spansize:
		reconstructed_sent = reconstructed_sent[:max_spansize]
	return reconstructed_sent


# This function calculates the maximum cosine similarity score.
def calc_per_entry_score_bert(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
							  bert_device, max_context_size=None, debug=False, is_wh=False):
	assert method in ['bert1A', 'bert2A', 'bert3A']
	# the data entry can be positive or negative, so that must be scenario 3; but the references here can be scenario 2
	query_sent = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=is_wh)
	query_toks = bert_tokenizer([query_sent], padding=True)
	query_toks = query_toks.convert_to_tensors('pt')
	query_toks = query_toks.to(bert_device)
	query_outputs = bert_model(**query_toks)
	query_vecs = query_outputs.last_hidden_state
	assert query_vecs.shape[0] == 1
	query_vecs = query_vecs[:, 0, :].cpu().detach().numpy()

	if max_context_size is not None and len(ref_sents) > max_context_size:
		assert method not in ['bert1A']
		# if debug:
		print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(ref_sents)}!")
		assert ref_rels is None or len(ref_rels) == len(ref_sents)
		sample_ids = random.sample(range(len(ref_sents)), k=max_context_size)
		new_ref_rels = [] if ref_rels is None else None
		new_ref_sents = []
		for i in sample_ids:
			new_ref_sents.append(ref_sents[i])
			if ref_rels is not None:
				new_ref_rels.append(ref_rels[i])

	ref_emb_inputstrs = []
	ref_answers = []
	ref_emb_outputvecs = []
	if method in ['bert2A', 'bert3A']:
		assert len(ref_rels) == len(ref_sents)
		for rrel, rsent in zip(ref_rels, ref_sents):

			if method == 'bert2A':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, rrel, max_spansize=max_spansize))
			elif method == 'bert3A':
				ref_emb_inputstrs.append(reconstruct_sent_from_rel(rrel, max_spansize))
			else:
				raise AssertionError
	else:
		for rsent in ref_sents:
			if method == 'bert1A':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, None, max_spansize=max_spansize))
			else:
				raise AssertionError

	ref_emb_chunks = []
	chunk_size = 32
	offset = 0
	while offset < len(ref_emb_inputstrs):
		ref_emb_chunks.append((offset, min(offset + chunk_size, len(ref_emb_inputstrs))))
		offset += chunk_size
	for chunk in ref_emb_chunks:
		if chunk[1] == chunk[0]:  # do not attempt to send empty input into the model!
			continue
		ref_emb_inputtoks = bert_tokenizer(ref_emb_inputstrs[chunk[0]:chunk[1]], padding=True)
		ref_emb_inputtoks = ref_emb_inputtoks.convert_to_tensors('pt')
		ref_emb_inputtoks = ref_emb_inputtoks.to(bert_device)
		ref_encoder_outputs = bert_model(**ref_emb_inputtoks)
		ref_encoder_outputs = ref_encoder_outputs.last_hidden_state
		if debug:
			print(ref_encoder_outputs.shape)
		ref_encoder_outputs = ref_encoder_outputs[:, 0, :].cpu().detach().numpy()
		for bidx in range(ref_encoder_outputs.shape[0]):
			ref_emb_outputvecs.append(ref_encoder_outputs[bidx, :])
	assert len(ref_emb_outputvecs) == len(ref_emb_inputstrs)
	ref_emb_outputvecs = np.array(ref_emb_outputvecs)

	# for wh-questions, the returned value is a dict, with answers as keys and corresponding similarities as values;
	# for boolean questions, the returned value is a float number, which is the maximum score.
	if is_wh:
		assert method not in ['bert1A']
		if len(ref_emb_inputstrs) > 0:
			cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
			assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
			cur_max_sim = np.amax(cur_sims)
			cur_argmax_sim = np.argmax(cur_sims)
			if debug:
				print(f"cur sims shape: {cur_sims.shape}")
				print(f"query rel: {query_ent['r']}")
				if ref_rels is not None:
					print(f"best ref rel: {ref_rels[cur_argmax_sim]['r']}")
				else:
					print(f"Best ref sent: {ref_sents[cur_argmax_sim]}")
		else:
			cur_max_sim = {}
			if debug:
				print(f"No relevant relations found!")
	else:
		if len(ref_emb_inputstrs) > 0:
			cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
			assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
			cur_max_sim = np.amax(cur_sims)
			cur_argmax_sim = np.argmax(cur_sims)
			if debug:
				print(f"cur sims shape: {cur_sims.shape}")
				print(f"query rel: {query_ent['r']}")
				if ref_rels is not None:
					print(f"best ref rel: {ref_rels[cur_argmax_sim]['r']}")
				else:
					print(f"Best ref sent: {ref_sents[cur_argmax_sim]}")
		else:
			cur_max_sim = 0.0
			if debug:
				print(f"No relevant relations found!")
	return cur_max_sim


# This function prepends the context to the question, and outputs softmaxed logits.
def in_context_prediction_bert(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
							  bert_device, max_context_size=None, debug=False, is_wh=False):
	assert is_wh is True
	assert method in ['bert1B', 'bert2B', 'bert3B']
	# the data entry can be positive or negative, so that must be scenario 3; but the references here can be scenario 2
	query_sent = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=is_wh)
	query_toks = bert_tokenizer([query_sent], padding=True)
	query_toks = query_toks.convert_to_tensors('pt')
	query_toks = query_toks.to(bert_device)
	query_outputs = bert_model(**query_toks)
	query_vecs = query_outputs.last_hidden_state
	assert query_vecs.shape[0] == 1
	query_vecs = query_vecs[:, 0, :].cpu().detach().numpy()

	if max_context_size is not None and len(ref_sents) > max_context_size:
		# if debug:
		print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(ref_sents)}!")
		assert ref_rels is None or len(ref_rels) == len(ref_sents)
		sample_ids = random.sample(range(len(ref_sents)), k=max_context_size)
		new_ref_rels = [] if ref_rels is None else None
		new_ref_sents = []
		for i in sample_ids:
			new_ref_sents.append(ref_sents[i])
			if ref_rels is not None:
				new_ref_rels.append(ref_rels[i])

	ref_emb_inputstrs = []
	ref_emb_outputvecs = []
	if method in ['bert2B', 'bert3B']:
		assert len(ref_rels) == len(ref_sents)
		for rrel, rsent in zip(ref_rels, ref_sents):
			if method == 'bert2B':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, rrel, max_spansize=max_spansize))
			elif method == 'bert3B':
				ref_emb_inputstrs.append(reconstruct_sent_from_rel(rrel, max_spansize))
			else:
				raise AssertionError
	else:
		for rsent in ref_sents:
			if method == 'bert1B':
				ref_emb_inputstrs.append(fetch_span_by_rel(rsent, None, max_spansize=max_spansize))
			else:
				raise AssertionError

	ref_emb_chunks = []
	chunk_size = 32
	offset = 0
	while offset < len(ref_emb_inputstrs):
		ref_emb_chunks.append((offset, min(offset + chunk_size, len(ref_emb_inputstrs))))
		offset += chunk_size
	for chunk in ref_emb_chunks:
		if chunk[1] == chunk[0]:  # do not attempt to send empty input into the model!
			continue
		ref_emb_inputtoks = bert_tokenizer(ref_emb_inputstrs[chunk[0]:chunk[1]], padding=True)
		ref_emb_inputtoks = ref_emb_inputtoks.convert_to_tensors('pt')
		ref_emb_inputtoks = ref_emb_inputtoks.to(bert_device)
		ref_encoder_outputs = bert_model(**ref_emb_inputtoks)
		ref_encoder_outputs = ref_encoder_outputs.last_hidden_state
		if debug:
			print(ref_encoder_outputs.shape)
		ref_encoder_outputs = ref_encoder_outputs[:, 0, :].cpu().detach().numpy()
		for bidx in range(ref_encoder_outputs.shape[0]):
			ref_emb_outputvecs.append(ref_encoder_outputs[bidx, :])
	assert len(ref_emb_outputvecs) == len(ref_emb_inputstrs)
	ref_emb_outputvecs = np.array(ref_emb_outputvecs)
	if len(ref_emb_inputstrs) > 0:
		cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
		assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
		cur_max_sim = np.amax(cur_sims)
		cur_argmax_sim = np.argmax(cur_sims)
		if debug:
			print(f"cur sims shape: {cur_sims.shape}")
			print(f"query rel: {query_ent['r']}")
			if ref_rels is not None:
				print(f"best ref rel: {ref_rels[cur_argmax_sim]['r']}")
			else:
				print(f"Best ref sent: {ref_sents[cur_argmax_sim]}")
	else:
		cur_max_sim = 0.0
		if debug:
			print(f"No relevant relations found!")
	return cur_max_sim


def find_entailment_matches_from_graph(_graph, ent, ref_rels, typematch_flag, feat_idx, debug=False):
	# find entailment matches for one entry from one graph
	maximum_tscore = None
	maximum_uscore = None
	max_tscore_ref = None
	max_uscore_ref = None
	q_upred, q_subj, q_obj, q_tsubj, q_tobj = parse_rel(ent)
	assert '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
	if q_tsubj == q_tobj:
		q_tsubj = q_tsubj + '_1'
		q_tobj = q_tobj + '_2'
	q_tpred_querytype = '#'.join([q_upred, q_tsubj, q_tobj])
	assert len(_graph.types) == 2
	q_tpred_graphtype_fwd = '#'.join([q_upred, _graph.types[0], _graph.types[1]])
	q_tpred_graphtype_rev = '#'.join([q_upred, _graph.types[1], _graph.types[0]])

	num_true_entailments = 0.0

	for rrel in ref_rels:
		r_upred, r_subj, r_obj, r_tsubj, r_tobj = parse_rel(rrel)
		assert '_1' not in r_tsubj and '_2' not in r_tsubj and '_1' not in r_tobj and '_2' not in r_tobj
		if r_tsubj == r_tobj:
			# the assertion below seems deprecated, the ref argument types are allowed to be different from the query
			# argument types, but in these cases the ref argument types would be ignored.
			# assert q_tsubj[:-2] == q_tobj[:-2] and '_1' in q_tsubj and '_2' in q_tobj
			if rrel['aligned'] is True:
				r_tsubj = r_tsubj + '_1'
				r_tobj = r_tobj + '_2'
			elif rrel['aligned'] is False:
				r_tsubj = r_tsubj + '_2'
				r_tobj = r_tobj + '_1'
			else:
				raise AssertionError
		else:
			# assert q_tsubj != q_tobj and '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
			pass

		if rrel['aligned'] is True:
			r_tpred_querytype = '#'.join([r_upred, q_tsubj, q_tobj])
			r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
			r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
		elif rrel['aligned'] is False:
			r_tpred_querytype = '#'.join([r_upred, q_tobj, q_tsubj])
			r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
			r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
		else:
			raise AssertionError

		# print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
		# TODO: Attention! Check this feat_idx matter!
		effective_feat_idx = feat_idx + 0
		if typematch_flag is True:
			assert (q_tsubj == _graph.types[0] and q_tobj == _graph.types[1]) or \
				   (q_tsubj == _graph.types[1] and q_tobj == _graph.types[0])

			cur_tscores = _graph.get_features(r_tpred_querytype, q_tpred_querytype)
			if cur_tscores is not None:
				# print(f"cur tscores length: {len(cur_tscores)}")
				# print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
				if cur_tscores[0] > 0 and cur_tscores[0] < 0.99:
					num_true_entailments += 1
				curfeat_cur_tscore = cur_tscores[effective_feat_idx]
				if maximum_tscore is None or curfeat_cur_tscore > maximum_tscore:
					max_tscore_ref = r_tpred_querytype
					maximum_tscore = curfeat_cur_tscore
		else:
			num_true_entailments = None

		cur_uscores_fwd = _graph.get_features(r_tpred_graphtype_fwd, q_tpred_graphtype_fwd)
		cur_uscores_rev = _graph.get_features(r_tpred_graphtype_rev, q_tpred_graphtype_rev)

		if cur_uscores_fwd is not None:
			if cur_uscores_fwd[1] > 0 and cur_uscores_fwd[1] < 0.99:
				# print(cur_uscores_fwd)
				# [0.43617837 0.2739726  0.16573886 0.18112025 0.13438165 0.09970408, 0.5        0.5        1.         0.5        1.         1.        ]
				# print("!")
				pass
			curfeat_cur_uscore_fwd = cur_uscores_fwd[effective_feat_idx]
			if maximum_uscore is None or curfeat_cur_uscore_fwd > maximum_uscore:
				max_uscore_ref = r_tpred_graphtype_fwd
				maximum_uscore = curfeat_cur_uscore_fwd
		if cur_uscores_rev is not None:
			curfeat_cur_uscore_rev = cur_uscores_rev[effective_feat_idx]
			if maximum_uscore is None or curfeat_cur_uscore_rev > maximum_uscore:
				max_uscore_ref = r_tpred_graphtype_rev
				maximum_uscore = curfeat_cur_uscore_rev

	if debug and maximum_tscore is not None:
		print(f"query: {q_tpred_querytype}; max_tscore_ref: {max_tscore_ref}; max_uscore_ref: {max_uscore_ref}")

	return maximum_tscore, maximum_uscore, num_true_entailments


def find_answers_from_graph(_graph, ent, ref_rels, typematch_flag, partial_typematch_flag, feat_idx, debug=False):
	raise NotImplementedError