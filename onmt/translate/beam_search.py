import torch

from collections import Counter
import copy

from onmt.translate.decode_strategy import DecodeStrategy



class BeamSearch(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        mb_device (torch.device or str): See base ``device``.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """

    def __init__(self, beam_size, batch_size, pad, bos, eos, n_best, mb_device,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens, memory_lengths,
                 stepwise_penalty, ratio, src_counter, src_tree_nodes, src):
        # import ipdb; ipdb.set_trace()
        # TODO make this optional
        # src_tokens = [tok not  for tok in src[:, 0, 0].tolist() if tok not in [4, 5]]
        # ignore the scoping brackets
        num_tokens = sum(tok not in [4, 5] for tok in src[:, 0, 0].tolist())

        min_length = num_tokens
        # # max_length = num_tokens
        # # ratio = 1
        self.src_counter = src_counter
        self.src_tree_nodes = src_tree_nodes
        self.conllu_ids_so_far = [[] for i in range(beam_size)]

        super(BeamSearch, self).__init__(
            pad, bos, eos, batch_size, mb_device, beam_size, min_length,
            block_ngram_repeat, exclusion_tokens, return_attention,
            max_length)
        # beam parameters
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        self.ratio = ratio

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        self.best_scores = torch.full([batch_size], -1e10, dtype=torch.float,
                                      device=mb_device)

        self._batch_offset = torch.arange(batch_size, dtype=torch.long)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long,
            device=mb_device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=mb_device
        ).repeat(batch_size)
        self.select_indices = None
        self._memory_lengths = memory_lengths

        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((batch_size, beam_size),
                                       dtype=torch.float, device=mb_device)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long,
                                    device=mb_device)
        self._batch_index = torch.empty([batch_size, beam_size],
                                        dtype=torch.long, device=mb_device)
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
                stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_origin(self):
        return self.select_indices

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    def restrict_repetition(self, log_probs):
        for path_idx in range(self.alive_seq.shape[0]):
            # skip BOS
            hyp = self.alive_seq[path_idx, 1:]
            this_counter = Counter(hyp.tolist())
            remaining_toks = self.src_counter - this_counter
            # make list of length vocab size
            disallowed_token_idxs = [True] * log_probs.size(-1)
            for key in remaining_toks:
                disallowed_token_idxs[key] = False
            # https://discuss.pytorch.org/t/slicing-tensor-using-boolean-list/7354/5
            disallowed_lookup = torch.Tensor(disallowed_token_idxs) == True
            log_probs[path_idx, disallowed_lookup] = -10e20
            # if any(this_counter - self.src_counter):
            #     log_probs[path_idx] = -10e20

    def mask_log_probs(self, log_probs, allowed_tokens, path_idx):
        disallowed_token_idxs = [True] * log_probs.size(-1)
        for key in allowed_tokens:
            disallowed_token_idxs[key] = False
        # EOS is masked later on
        disallowed_token_idxs[3] = False
        # https://discuss.pytorch.org/t/slicing-tensor-using-boolean-list/7354/5
        disallowed_lookup = torch.Tensor(disallowed_token_idxs) == True
        log_probs[path_idx, disallowed_lookup] = -10e20


    def depth_first_decoding(self, log_probs):
        # We're assuming that none of the beams will die before all tokens are
        # generated and that they will all die at the same time. We can do this
        # because we know exactly how many tokens there are and what specific
        # tokens will appear
        for path_idx in range(self.alive_seq.shape[0]):
            current_step = self.alive_seq.size(-1)
            if current_step == 1:
                allowed_tokens = {self.src_tree_nodes['1'].tok_idx: '1'}
                self.mask_log_probs(log_probs, allowed_tokens, path_idx)
            elif current_step == 2:
                self.conllu_ids_so_far[path_idx].append('1')
                allowed_tokens = self.get_allowed_idxs(
                    self.conllu_ids_so_far[path_idx])
                self.mask_log_probs(log_probs, allowed_tokens, path_idx)
            elif current_step > 2:
                # We were having trouble with beams changes order so we need to
                # keep track of what a beams history was in terms of conllu ids
                # drop the most recent choice so we can match it
                tok_history = self.alive_seq[path_idx, :-1]
                # There has to be a more pytorch-y way of doing this
                matching_rows = [
                    i for i in range(self.alive_seq.shape[0])
                    if all(tok_history == self.prev_alive_seq[i])
                ]
                self.conllu_ids_so_far[path_idx] = copy.copy(
                    self.pre_conllu_ids_so_far[matching_rows[0]])
                previously_allowed_tokens = self.get_allowed_idxs(
                        self.conllu_ids_so_far[path_idx])
                # we need to make sure this is converted to an int
                chosen_hyp = self.alive_seq[path_idx, -1].tolist()
                try:
                    chosen_conllu_id = previously_allowed_tokens[chosen_hyp]
                    self.conllu_ids_so_far[path_idx].append(chosen_conllu_id)
                except:
                    # We expect it to fail but still not sure why
                    # print('well ok it did that thing')
                    # print(self.src_tree_nodes)
                    pass
                allowed_tokens = self.get_allowed_idxs(
                        self.conllu_ids_so_far[path_idx])
                self.mask_log_probs(log_probs, allowed_tokens, path_idx)
        # make copies of the previous variables
        self.prev_alive_seq = self.alive_seq.clone()
        self.pre_conllu_ids_so_far = copy.deepcopy(self.conllu_ids_so_far)
        # print(log_probs)
        # print(log_probs.exp())
        # print(self.alive_seq)
        # print(self.conllu_ids_so_far)
        # import ipdb; ipdb.set_trace()
        # pass


    def get_allowed_idxs(self, so_far):
        idx = 0
        # Set default parent to root node as sometimes there are no children
        current_parent = self.src_tree_nodes['1']
        done_labels = {}
        while idx < len(so_far):
            label = so_far[idx]
            node = self.src_tree_nodes[label]
            if node.desc_count < len(so_far) - idx:
                done_labels[label] = True
                idx = idx + node.desc_count + 1
            else:
                current_parent = node
                idx = idx + 1
        allowed_nodes_mapping = {}
        # Note: If two child nodes have the same token vocab index then it's
        # impossible to tell which was chosen by the model. So we just have to
        # do it randomly. We still have to check how frequently this occurs but
        # it ought to be rare.
        for child_node in current_parent.children:
            this_label = child_node.name
            if this_label in done_labels:
                continue
            allowed_nodes_mapping[child_node.tok_idx] = this_label
        return allowed_nodes_mapping


    def advance(self, log_probs, attn):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        if self._stepwise_cov_pen and self._prev_penalty is not None:
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        self.block_ngram_repeats(log_probs)
        # TODO make this optional because it's redundant when using tree node
        # blocking
        self.restrict_repetition(log_probs)
        # import ipdb; ipdb.set_trace()
        self.depth_first_decoding(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)
        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage.index_select(
                        1, self.select_indices)
                    self._coverage += current_attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta).view(
                            _B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage,
                beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self._memory_lengths[i]]
                    if attention is not None else None))
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self._memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)
            if self._cov_pen:
                self._coverage = self._coverage \
                    .view(1, _B_old, self.beam_size, inp_seq_len) \
                    .index_select(1, non_finished) \
                    .view(1, _B_new * self.beam_size, inp_seq_len)
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty.index_select(
                        0, non_finished)

