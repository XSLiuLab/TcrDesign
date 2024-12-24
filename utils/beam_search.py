import torch
import torch.functional as F

import warnings
warnings.filterwarnings('ignore')

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1 
        self.length_penalty = length_penalty
        self.num_beams = num_beams 
        self.beams = [] 
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def beam_search(num_beams, max_length, vocab, decoder, has_encoder=False, encoder_input=None, vocab_size=None,
                decoder_input_prefix=None, with_cuda=True):
    eos_token_id = vocab.eos_index
    sos_token_id = vocab.sos_index
    pad_token_id = vocab.pad_index
    if vocab_size is None:
        vocab_size = len(vocab)
    cur_len = 1
    if encoder_input is not None:
        batch_size = encoder_input.size(0)
    
    beam_scores = torch.zeros((batch_size, num_beams))
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)
    done = [False for _ in range(batch_size)]
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty=0.7)
            for _ in range(batch_size)
    ]
    if decoder_input_prefix is None:
        input_ids =  torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long)
    else:
        input_ids =  decoder_input_prefix.repeat(1, num_beams).view(-1, decoder_input_prefix.size(1))
    encoder_input = encoder_input.repeat(1, num_beams).view(-1, encoder_input.size(1)) if has_encoder else None
    
    while cur_len < max_length:
        if with_cuda:
            encoder_input_cuda = encoder_input.cuda()
            input_ids_cuda = input_ids.cuda()
            with torch.no_grad():
                outputs = decoder(encoder_input_cuda, input_ids_cuda) # 输出为对数概率
            outputs = outputs.cpu()
            del encoder_input_cuda, input_ids_cuda
            torch.cuda.empty_cache() # 释放显存
        else:
            outputs = decoder(encoder_input, input_ids) # 输出为对数概率
        
        next_token_logits = outputs[:, -1, :]
        scores = next_token_logits
        next_scores = scores + beam_scores[:, None].expand_as(scores)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        ) 
        next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)
  
        next_batch_beam = []

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)
                continue
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size # 1
                token_id = beam_token_id % vocab_size # 1
                effective_beam_id = batch_idx * num_beams + beam_id
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                if len(next_sent_beam) == num_beams:
                    break
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len)
            next_batch_beam.extend(next_sent_beam)
            
        if all(done):
            break
        
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        input_ids = input_ids[beam_idx, :]
        encoder_input = encoder_input[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1
    
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    
    output_num_return_sequences_per_batch = num_beams
    output_batch_size = output_num_return_sequences_per_batch * batch_size
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        decoded = torch.stack(best).type(torch.long)

    return decoded
