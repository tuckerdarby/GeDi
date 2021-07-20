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
            print('Gedi model path taken')
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
                    inputs = {"input_ids": seq_batched, "past_key_values":gedi_past}
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
