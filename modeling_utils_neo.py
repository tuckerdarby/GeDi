import math
import torch
from torch.nn import functional as F
from typing import Optional, Union
from transformers import generation_utils, logging
import warnings
from transformers.file_utils import ModelOutput
from transformers.generation_utils import (
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
)

logger = logging.get_logger(__name__)

def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
           


class GediMixin(generation_utils.GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        gedi_model = None,
        rep_penalty_scale=0,
        penalize_cond=False,
        tokenizer=None,
        disc_weight=0,
        filter_p=1,
        target_p=1,
        class_bias=0,
        attr_class=0,
        code_0="negative",
        code_1="positive",
        multi_code=None,
        get_ll=False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to :obj:`model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (:obj:`int`, `optional`, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                :obj:`max_new_tokens` or :obj:`max_length` but not both, they serve the same purpose.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(:obj:`float`, `optional`, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID :obj:`batch_id` and
                :obj:`input_ids`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the batch ID :obj:`batch_id` and the previously generated tokens :obj:`inputs_ids`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            remove_invalid_values (:obj:`bool`, `optional`):
                Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
                crash. Note that using ``remove_invalid_values`` can slow down generation.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`

        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input context
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """

        # set init values
        if max_length is None and max_new_tokens is None:
            # Both are None, default
            max_length = self.config.max_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set but they serve the same purpose.", UserWarning
            )

        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        cur_len = input_ids.shape[-1]

        if gedi_model is not None:
            # Batch Size
            batch_size = 1
            if input_ids is not None:
                batch_size = input_ids.shape[0]  # overriden by the input batch_size                

            # Effective Batch Size
            if num_return_sequences != 1:
                # Expand input to num return sequences
                input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
                input_ids = input_ids.contiguous().view(
                    batch_size * num_return_sequences, cur_len
                )  # (batch_size * num_return_sequences, cur_len)
            
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                min_length,
                do_sample,
                no_repeat_ngram_size,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                rep_penalty_scale,
                pad_token_id,
                eos_token_id,
                batch_size,
                penalize_cond,
                gedi_model,
                tokenizer,
                disc_weight,
                filter_p,
                target_p,
                class_bias,
                attr_class,
                code_0,
                code_1,
                multi_code,
                get_ll
            )

            if num_return_sequences != 1:
                output = output.view(batch_size, num_return_sequences, -1)
            
            return output
        return None
  
    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        no_repeat_ngram_size,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        rep_penalty_scale,
        pad_token_id,
        eos_token_id,
        batch_size,
        penalize_cond,
        gedi_model,
        tokenizer,
        disc_weight,
        filter_p,
        target_p,
        class_bias,
        attr_class,
        code_0,
        code_1,
        multi_code,
        get_ll
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        #set this to 0 if you want to apply repetition_penalty to the prompt too
        if penalize_cond:
            cond_len = 0
        else:
            cond_len = input_ids.shape[1]

        if not(gedi_model is None):
            if attr_class == 0:
                pt_id = tokenizer.encode(code_0)[0]
                nt_id = tokenizer.encode(code_1)[0]
            elif attr_class == 1:
                nt_id = tokenizer.encode(code_0)[0]
                pt_id = tokenizer.encode(code_1)[0]
            else:
                raise RuntimeError("expects attr_class is 0 or 1")

            #prepending tokens corresponding to 'positive' and 'negative' to the inputs
            seq_a = (torch.ones(input_ids.shape[0])*pt_id).type_as(input_ids).view(-1,1)
            seq_b = (torch.ones(input_ids.shape[0])*nt_id).type_as(input_ids).view(-1,1)
            if not(multi_code is None):
                seq_a2 = torch.LongTensor(multi_code).unsqueeze(0).to(seq_a.device)
                seq_a = torch.cat((seq_a, seq_a2, input_ids), dim=1)[:,:]
                seq_b = torch.cat((seq_b, seq_a2, input_ids), dim=1)[:,:]

            else:

                seq_a = torch.cat((seq_a, input_ids), dim=1)[:,:]
                seq_b = torch.cat((seq_b, input_ids), dim=1)[:,:]

            seq_batched = torch.cat((seq_a,seq_b),dim=0)

        past = None
        gedi_past = None

        if get_ll:
            sequence_ll = 0

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if get_ll:
                next_token_logp = torch.log_softmax(next_token_logits,-1)
            if not(gedi_model is None):
                #want to compute LM loss here so feeding inputs as labels
                if not gedi_past is None:
                    input_batched = torch.cat((model_inputs["input_ids"],model_inputs["input_ids"]),dim=0)
                    seq_batched = torch.cat((seq_batched,input_batched),dim=1)
                    inputs = gedi_model.prepare_inputs_for_generation(seq_batched, past=gedi_past)
                else:
                    inputs = {"input_ids": seq_batched, "past":gedi_past}
                gedi_outputs = gedi_model(**inputs)
                if gedi_past is None:
                    if gedi_outputs[0].shape[1]>1:
                        shift_logits = gedi_outputs[0][..., :-1, :].contiguous()
                        shift_labels = seq_batched[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        logits_r  = -1*loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        logits_r = logits_r.view(seq_batched.shape[0], -1)

                        seq_len = logits_r.shape[1]

                        logits_r = torch.sum(logits_r,1)
                        logits_pos,logits_neg = torch.split(logits_r/seq_len,input_ids.shape[0])
                        logits0 = torch.stack((logits_pos,logits_neg),1)

                        if "logit_scale" in dir(gedi_model):
                            logits0 = gedi_model.logit_scale*logits0

                        if "bias" in dir(gedi_model):
                            logits0 = logits0 + gedi_model.bias
                        if not (class_bias==0):
                            logits0[:,0] += class_bias


                        logp_desired = torch.log_softmax(logits0,-1)[:,0]
                        logp_undesired = torch.log_softmax(logits0,-1)[:,1]
                    else:
                        seq_len=0
                        logp_desired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
                        logp_undesired = (torch.zeros(input_ids.shape[0]) + torch.log(torch.tensor(0.5))).to(input_ids.device)
                        logits_r = torch.zeros(input_ids.shape[0]*2).to(input_ids.device)


                seq_len= seq_len+1
                gedi_logits= (torch.log_softmax(gedi_outputs[0][:, -1, :],-1)+logits_r.unsqueeze(1))

                logits_pos,logits_neg = torch.split(gedi_logits/seq_len,input_ids.shape[0])
                logits = torch.stack((logits_pos,logits_neg),2)
                if "logit_scale" in dir(gedi_model):
                    logits = gedi_model.logit_scale*logits

                if "bias" in dir(gedi_model):
                    logits = logits + gedi_model.bias

                if not class_bias == 0:
                    logits[:,:,0] += class_bias

                logp_desired_t = torch.log_softmax(logits,-1)[:,:,0]
                logp_undesired_t = torch.log_softmax(logits,-1)[:,:,1]

                next_token_logits = torch.log_softmax(1*next_token_logits,-1) + disc_weight*(logp_desired_t) #+delta_capped82058721

                _, sorted_indices = torch.sort(logp_desired_t, descending=False)

                next_token_p = torch.softmax(next_token_logits,-1)
                for i in range(0,next_token_logits.shape[0]):


                    if True:
                        p_sorted = next_token_p[i,sorted_indices[i]]
                        cumulative_probs = torch.cumsum(p_sorted, dim=-1)

                        logp_desired_sorted = logp_desired_t[i,sorted_indices[i]]


                        ind_to_remove =  (cumulative_probs <filter_p)  & (logp_desired_sorted<(math.log(target_p)))

                        next_token_logits[i,sorted_indices[i][ind_to_remove]]-=10000



                        if ind_to_remove[-1]:
                            print("error, removing everything is likely not intended behavior")
                            ind_to_remove[-1]=True

            # if model has past, then set the past variable to speed up decoding

            if not (gedi_model is None):
                gedi_past = gedi_outputs[1]

            max = torch.max(next_token_logits,-1,keepdim=True)
            max=max[0]
            next_token_logits= next_token_logits - max + rep_penalty_scale

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):

                    prevs = input_ids[i][cond_len:].tolist()

                    for j in range(0,len(prevs)):
                        previous_token = prevs[j]

                        if rep_penalty_scale>0:
                            if  next_token_logits[i, previous_token] == rep_penalty_scale:
                                rescale=True
                            else:
                                rescale=False

                            next_token_logits[i, previous_token] /= repetition_penalty
                            #original version accidentally put rescaling inside forloop over prevs, this is slow and only changes things is max logit is penalized
                            #conditonal replicates paper results but is faster
                            #can comment out to remove, makes very small difference, generation sometimes the same
                            if rescale:

                                max = torch.max(next_token_logits[i,:])
                                next_token_logits[i,:]= next_token_logits[i,:]- max + rep_penalty_scale
                        else:
                            if next_token_logits[i, previous_token] < 0:
                                next_token_logits[i, previous_token] *= repetition_penalty
                            else:
                                next_token_logits[i, previous_token] /= repetition_penalty

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids[cond_len:], batch_size, no_repeat_ngram_size, len(input_ids[cond_len:]))
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if not(gedi_model is None):
                for i in range(batch_size):
                    if (cur_len < min_length):
                        next_token_logits[i, eos_token_id] -=10000

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)


            if not (gedi_model is None):
                token_list = next_token.tolist()+next_token.tolist()

                for i in range(0,len(token_list)):
                    logits_r[i] = gedi_logits[i,token_list[i]]

                for i in range(0,len(next_token)):
                    logp_desired[i] = logp_desired_t[i,next_token[i]]
                    logp_undesired[i] = logp_undesired_t[i,next_token[i]]


            # update generations and finished sentences
            tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
            if get_ll:
                sequence_ll += next_token_logp[0,next_token]
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            unfinished_sents.mul_(tokens_to_add.ne([eos_token_id]).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        if not(gedi_model is None):
            print("GeDi estimates the probability that it sample is desired class is: " + str(torch.exp(logp_desired[0]).item()))

        # add eos_token_ids to unfinished sentences

        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(unfinished_sents.to(dtype=torch.bool), eos_token_id)

        if get_ll:
            return input_ids,sequence_ll
        else:
            return input_ids
