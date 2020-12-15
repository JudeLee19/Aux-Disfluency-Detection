import torch
import torch.nn as nn
import numpy as np
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0 ] +torch.log(torch.exp(log_M -torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0 ] +torch.log \
        (torch.exp(log_Tensor -torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0] ,-1 ,1)).sum(axis))


class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id,
                 num_labels, batch_size, device,
                 num_ner_labels=None, ner_start_label_id=None, ner_stop_label_id=None,
                 num_pos_labels=None, pos_start_label_id=None, pos_stop_label_id=None,
                 language_model='bert'):

        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.num_ner_labels = num_ner_labels
        self.num_pos_labels = num_pos_labels
        self.batch_size = batch_size
        self.device =device
        self.language_model = language_model

        self.ner_start_label_id = ner_start_label_id
        self.ner_stop_label_id = ner_stop_label_id

        self.pos_start_label_id = pos_start_label_id
        self.pos_stop_label_id = pos_stop_label_id

        print('num_ner_labels: ', self.num_ner_labels)
        print('num_pos_labels: ', self.num_pos_labels)

        # use pretrainded BertModel
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)
        self.hidden2_ner_label = nn.Linear(self.hidden_size, self.num_ner_labels)
        self.hidden2_pos_label = nn.Linear(self.hidden_size, self.num_pos_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        if num_ner_labels:
            self.ner_transitions = nn.Parameter(
                torch.randn(self.num_ner_labels, self.num_ner_labels))
            self.ner_transitions.data[ner_start_label_id, :] = -10000
            self.ner_transitions.data[:, ner_stop_label_id] = -10000

        if num_pos_labels:
            self.pos_transitions = nn.Parameter(
                torch.randn(self.num_pos_labels, self.num_pos_labels))
            self.pos_transitions.data[pos_start_label_id, :] = -10000
            self.pos_transitions.data[:, pos_stop_label_id] = -10000

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

        nn.init.xavier_uniform_(self.hidden2_ner_label.weight)
        nn.init.constant_(self.hidden2_ner_label.bias, 0.0)

        nn.init.xavier_uniform_(self.hidden2_pos_label.weight)
        nn.init.constant_(self.hidden2_pos_label.bias, 0.0)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats, type='dis'):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)

        if type == 'dis':
            log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
            # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
            # self.start_label has all of the score. it is log,0 is p=1
            log_alpha[:, 0, self.start_label_id] = 0

            # feats: sentances -> word embedding -> lstm -> MLP -> feats
            # feats is the probability of emission, feat.shape=(1,tag_size)
            for t in range(1, T):
                log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        elif type == 'ner':
            log_alpha = torch.Tensor(batch_size, 1, self.num_ner_labels).fill_(-10000.).to(self.device)
            # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
            # self.start_label has all of the score. it is log,0 is p=1
            log_alpha[:, 0, self.ner_start_label_id] = 0

            # feats: sentances -> word embedding -> lstm -> MLP -> feats
            # feats is the probability of emission, feat.shape=(1,tag_size)
            for t in range(1, T):
                log_alpha = (log_sum_exp_batch(self.ner_transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        elif type == 'pos':
            log_alpha = torch.Tensor(batch_size, 1, self.num_pos_labels).fill_(-10000.).to(self.device)
            # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
            # self.start_label has all of the score. it is log,0 is p=1
            log_alpha[:, 0, self.pos_start_label_id] = 0

            # feats: sentances -> word embedding -> lstm -> MLP -> feats
            # feats is the probability of emission, feat.shape=(1,tag_size)
            for t in range(1, T):
                log_alpha = (log_sum_exp_batch(self.pos_transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask,
                           use_ner=None, use_pos=None):
        '''
        sentances -> word embedding -> lstm -> MLP -> feats
        '''
        if self.language_model == 'bert':
            bert_seq_out, _ = self.bert(input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        output_all_encoded_layers=False)

        elif self.language_model == 'electra':
            bert_seq_out = self.bert(input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask, return_dict=True)

            bert_seq_out = bert_seq_out.last_hidden_state
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        bert_ner_feats = None
        bert_pos_feats = None
        if use_ner:
            bert_ner_feats = self.hidden2_ner_label(bert_seq_out)
        if use_pos:
            bert_pos_feats = self.hidden2_pos_label(bert_seq_out)
        return bert_feats, bert_ner_feats, bert_pos_feats

    def _score_sentence(self, feats, label_ids, type='dis'):
        '''
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        if type == 'dis':
            batch_transitions = self.transitions.expand(batch_size ,self.num_labels ,self.num_labels)
            batch_transitions = batch_transitions.flatten(1)

            score = torch.zeros((feats.shape[0] ,1)).to(self.device)
            # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
            for t in range(1, T):
                score = score + \
                        batch_transitions.gather(-1, (label_ids[:, t ] *self.num_labels +label_ids[:, t- 1]).view(-1, 1)) \
                        + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        elif type == 'ner':
            batch_transitions = self.ner_transitions.expand(batch_size, self.num_ner_labels, self.num_ner_labels)
            batch_transitions = batch_transitions.flatten(1)

            score = torch.zeros((feats.shape[0], 1)).to(self.device)
            # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
            for t in range(1, T):
                score = score + \
                        batch_transitions.gather(-1,
                                                 (label_ids[:, t] * self.num_ner_labels + label_ids[:, t - 1]).view(-1, 1)) \
                        + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)

        elif type == 'pos':
            batch_transitions = self.pos_transitions.expand(batch_size, self.num_pos_labels, self.num_pos_labels)
            batch_transitions = batch_transitions.flatten(1)

            score = torch.zeros((feats.shape[0], 1)).to(self.device)
            # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
            for t in range(1, T):
                score = score + \
                        batch_transitions.gather(-1,
                                                 (label_ids[:, t] * self.num_pos_labels + label_ids[:, t - 1]).view(-1, 1)) \
                        + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)

        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids,
                           ner_label_ids=None, pos_label_ids=None,
                           ner_gomma=None, pos_gomma=None):
        bert_feats, bert_ner_feats, bert_pos_feats = self._get_bert_features(input_ids, segment_ids, input_mask,
                                                                             use_ner=True, use_pos=True)
        forward_score = self._forward_alg(bert_feats, type='dis')
        gold_score = self._score_sentence(bert_feats, label_ids, type='dis')

        ner_forward_score = self._forward_alg(bert_ner_feats, type='ner')
        ner_gold_score = self._score_sentence(bert_ner_feats, ner_label_ids, type='ner')

        pos_forward_score = self._forward_alg(bert_pos_feats, type='pos')
        pos_gold_score = self._score_sentence(bert_pos_feats, pos_label_ids, type='pos')

        dis_loss = forward_score - gold_score
        ner_loss = ner_forward_score - ner_gold_score
        pos_loss = pos_forward_score - pos_gold_score

        loss = dis_loss + (ner_gomma * ner_loss) + (pos_gomma * pos_loss)

        print('ner_gomma: ', ner_gomma, 'ner_loss: ', torch.mean(ner_loss))
        print('pos_gomma: ', pos_gomma, 'pos_loss: ', torch.mean(pos_loss))

        return torch.mean(loss)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats, _, _ = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids