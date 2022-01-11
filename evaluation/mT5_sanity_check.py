import transformers
from transformers import T5Tokenizer, MT5ForConditionalGeneration

mt5_tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
mt5_model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

sents = ['咽 炎 是 <extra_id_0> 的 原 因 。 </s>',
		'咽 炎 不 会 导 致 肺 炎 。 咽 炎 可 能 造 成 肿 痛 。 咽 炎 需 要 治 疗  。 西 瓜 霜 治 疗 咽 炎 。 咽 炎 是 <extra_id_0> 的 原 因 。 </s>',
		'咽 炎 不 会 引 起 发 烧 。 咽 炎 导 致 肿 痛 。 西 瓜 霜 治 疗 咽 炎 。 咽 炎 是 <extra_id_0> 的 原 因 。 </s>',
		 '印度是<extra_id_0>的国家。',
		 '印 度 是 <extra_id_0> 的 国 家 。 </s>',
		'中 国 是 <extra_id_0> 的 国 家 。 </s>',
		'美 国 是 <extra_id_0> 的 国 家 。 </s>',
		'法 国 是 <extra_id_0> 的 国 家 。 </s>',
		 'China is a <extra_id_0> of the world. </s>',
		 'United States is a <extra_id_0> of the world. </s>',
		 'France is a <extra_id_0> of the world. </s>']

# sent = 'China is a <extra_id_0> of the world. </s>'

for sent in sents:
	encoded = mt5_tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt')
	input_ids = encoded['input_ids']

	outputs = mt5_model.generate(input_ids=input_ids, num_beams=200, num_return_sequences=50, max_length=10)

	_0_index = sent.index('<extra_id_0>')
	_result_prefix = sent[:_0_index]
	_result_suffix = sent[_0_index+12:]  # 12 is the length of <extra_id_0>

	end_token = '<extra_id_1>'
	for op in outputs[:10]:
		result = mt5_tokenizer.decode(op[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)

		if end_token in result:
			_end_token_idx = result.index(end_token)
			result = result[:_end_token_idx]
		else:
			pass
		print(_result_prefix+'【'+result+'】'+_result_suffix)
	print("")