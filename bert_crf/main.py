import argparse
import torch
from torch.utils import data
import collections
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import ElectraModel, ElectraTokenizer
from data_utils import *
from models import *
from collections import defaultdict
import time
import os


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def train(hparams):
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")

    batch_size = hparams.batch_size
    gradient_accumulation_steps = hparams.gradient_accumulation_steps
    total_train_epochs = hparams.total_train_epochs
    bert_model_scale = hparams.bert_model_scale
    do_lower_case = hparams.do_lower_case
    max_seq_length = hparams.max_seq_length
    load_checkpoint = hparams.load_checkpoint
    output_dir = hparams.output_dir
    weight_decay_crf_fc =hparams.weight_decay_crf_fc
    learning_rate0 = hparams.learning_rate0
    warmup_proportion = hparams.warmup_proportion
    weight_decay_finetune = hparams.weight_decay_finetune
    lr0_crf_fc = hparams.lr0_crf_fc
    gomma_ner_loss = hparams.gomma_ner_loss
    gomma_pos_loss = hparams.gomma_pos_loss
    language_model = hparams.language_model
    electra_model_scale = hparams.electra_model_scale

    # Load pre-trained model tokenizer (vocabulary)
    swbdProcessor = SwbdDataProcessor(hparams.tagger_type)
    label_list = swbdProcessor.get_labels()
    label_map = swbdProcessor.get_label_map()
    ner_label_list = swbdProcessor.get_ner_labels()
    ner_label_map = swbdProcessor.get_ner_label_map()

    pos_label_list = swbdProcessor.get_pos_labels()
    pos_label_map = swbdProcessor.get_pos_label_map()

    print('NER Tagger Type: ', hparams.tagger_type)
    print('ner_label_list: ', ner_label_list)
    print('ner_label_map: ',  ner_label_map)
    print('pos_label_list: ', pos_label_list)
    print('pos_label_map:', pos_label_map)

    train_examples = swbdProcessor.get_train_examples(data_dir, hparams.tagger_type)
    dev_examples = swbdProcessor.get_dev_examples(data_dir, hparams.tagger_type)

    total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_examples))
    print("  Batch size = %d" % batch_size)
    print("  Num steps = %d" % total_train_steps)

    if language_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(bert_model_scale,
                                                  do_lower_case=do_lower_case)
    elif language_model == 'electra':
        tokenizer = ElectraTokenizer.from_pretrained(electra_model_scale,
                                                  do_lower_case=do_lower_case)


    train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length,
                               ner_label_map=ner_label_map, pos_label_map=pos_label_map)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length,
                             ner_label_map=ner_label_map, pos_label_map=pos_label_map)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     collate_fn=NerDataset.pad)


    start_label_id = swbdProcessor.get_start_label_id()
    stop_label_id = swbdProcessor.get_stop_label_id()

    start_ner_label_id = swbdProcessor.get_start_ner_label_id()
    stop_ner_label_id = swbdProcessor.get_stop_ner_label_id()

    start_pos_label_id = swbdProcessor.get_start_pos_label_id()
    stop_pos_label_id = swbdProcessor.get_stop_pos_label_id()

    if language_model == 'bert':
        lm_model = BertModel.from_pretrained(bert_model_scale)
    elif language_model == 'electra':
        lm_model = ElectraModel.from_pretrained(electra_model_scale)

    model = BERT_CRF_NER(lm_model, start_label_id, stop_label_id, len(label_list), batch_size, device,
                         num_ner_labels=len(ner_label_list),
                         ner_start_label_id=start_ner_label_id,
                         ner_stop_label_id=stop_ner_label_id,
                         num_pos_labels=len(pos_label_list),
                         pos_start_label_id=start_pos_label_id,
                         pos_stop_label_id=stop_pos_label_id,
                         language_model=language_model
                         )

    # %%
    if load_checkpoint and os.path.exists(output_dir + '/ner_bert_crf_checkpoint.pt'):
        checkpoint = torch.load(output_dir + '/ner_bert_crf_checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_CRF model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
            , 'lr': lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
            , 'lr': lr0_crf_fc, 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion,
                         t_total=total_train_steps)
    global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

    # train_start=time.time()
    # for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
    for epoch in range(start_epoch, total_train_epochs):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids, ner_label_ids, pos_label_ids = batch

            neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids,
                                                          ner_label_ids=ner_label_ids, pos_label_ids=pos_label_ids,
                                                          ner_gomma=float(gomma_ner_loss),
                                                          pos_gomma=float(gomma_pos_loss))

            if gradient_accumulation_steps > 1:
                neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

            neg_log_likelihood.backward()

            tr_loss += neg_log_likelihood.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader),
                                                                       neg_log_likelihood.item()))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        valid_precision,valid_recall, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

        # Save a checkpoint
        if valid_f1 > valid_f1_prev:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),  'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                       os.path.join(output_dir, 'ner_bert_crf_checkpoint.pt'))
            valid_f1_prev = valid_f1


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids, ner_label_ids, pos_label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            # _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')

    precision = round(100.*precision, 2)
    recall = round(100.*recall, 2)
    f1 = round(100.*f1, 2)
    return precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--eperiment_name', help='experiment_name', default='None')
    parser.add_argument('-ng', '--gomma_ner_loss', help='gomma_ner_loss', default=0.0)
    parser.add_argument('-pg', '--gomma_pos_loss', help='gomma_pos_loss', default=0.0)
    parser.add_argument('-t', '--ner_tagger', help='ner_tag_type', default=None)
    parser.add_argument('-m', '--mode', default='train')
    parser.add_argument('-lm', '--language_model', default='bert')
    parser.add_argument('-lan', '--language', default='english')
    parser.add_argument('-s', '--seed', default=44)


    args = parser.parse_args()

    if args.language == 'english':
        data_dir = os.path.join('../data/swbd/with_hand_craft_ner/')
        hparams = defaultdict(
            # "Whether to run training."
            do_train=True,
            # "Whether to run eval on the dev set."
            do_eval=True,
            # "Whether to run the model in inference mode on the test set."
            do_predict=True,
            # Whether load checkpoint file before train model
            load_checkpoint=True,
            # "The vocabulary file that the BERT model was trained on."
            max_seq_length=180,  # 256
            batch_size=32,  # 32
            # "The initial learning rate for Adam."
            learning_rate0=5e-5,
            lr0_crf_fc=8e-5,
            weight_decay_finetune=1e-5,  # 0.01
            weight_decay_crf_fc=5e-6,  # 0.005
            total_train_epochs=20,
            gradient_accumulation_steps=1,
            warmup_proportion=0.1,
            output_dir='../models/' + args.eperiment_name,
            bert_model_scale='bert-base-cased',
            electra_model_scale='google/electra-base-discriminator',
            do_lower_case=False,
            tagger_type=args.ner_tagger,
            gomma_ner_loss=args.gomma_ner_loss,
            gomma_pos_loss=args.gomma_pos_loss,
            language_model=args.language_model,
            seed=int(args.seed)
        )
    elif args.language == 'korean':
        data_dir = os.path.join('../data/korean/with_hand_craft_ner/')
        hparams = defaultdict(
            # "Whether to run training."
            do_train=True,
            # "Whether to run eval on the dev set."
            do_eval=True,
            # "Whether to run the model in inference mode on the test set."
            do_predict=True,
            # Whether load checkpoint file before train model
            load_checkpoint=True,
            # "The vocabulary file that the BERT model was trained on."
            max_seq_length=256,  # 256
            batch_size=32,  # 32
            # "The initial learning rate for Adam."
            learning_rate0=5e-5,
            lr0_crf_fc=5e-4,
            weight_decay_finetune=1e-5,  # 0.01
            weight_decay_crf_fc=5e-6,  # 0.005
            total_train_epochs=40,
            gradient_accumulation_steps=1,
            warmup_proportion=0.1,
            output_dir='../korean_models/' + args.eperiment_name,
            bert_model_scale='/data/project/rw/jude/data/word_embedding/bert.bpe.4.8m_step/',
            electra_model_scale='google/electra-base-discriminator',
            do_lower_case=False,
            tagger_type=args.ner_tagger,
            gomma_ner_loss=args.gomma_ner_loss,
            gomma_pos_loss=args.gomma_pos_loss,
            language_model=args.language_model,
            seed=int(args.seed)
        )

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    print('hparams:', hparams)

    if not os.path.exists(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    print('NER Tagger Type: ', args.ner_tagger)

    if args.mode == 'train':
        print('============ Train Mode ============')
        train(hparams)
    else:
        print('============ Evaluation Mode ============')
        swbdProcessor = SwbdDataProcessor(hparams.tagger_type)
        start_label_id = swbdProcessor.get_start_label_id()
        stop_label_id = swbdProcessor.get_stop_label_id()
        label_list = swbdProcessor.get_labels()
        label_map = swbdProcessor.get_label_map()

        # for ner settings
        ner_label_list = swbdProcessor.get_ner_labels()
        ner_label_map = swbdProcessor.get_ner_label_map()
        start_ner_label_id = swbdProcessor.get_start_ner_label_id()
        stop_ner_label_id = swbdProcessor.get_stop_ner_label_id()

        pos_label_list = swbdProcessor.get_pos_labels()
        pos_label_map = swbdProcessor.get_pos_label_map()
        start_pos_label_id = swbdProcessor.get_start_pos_label_id()
        stop_pos_label_id = swbdProcessor.get_stop_pos_label_id()

        dev_examples = swbdProcessor.get_dev_examples(data_dir, hparams.tagger_type)
        test_examples = swbdProcessor.get_test_examples(data_dir, hparams.tagger_type)

        if hparams.language_model == 'bert':
            tokenizer = BertTokenizer.from_pretrained(hparams.bert_model_scale,
                                                      do_lower_case=hparams.do_lower_case)
        elif hparams.language_model == 'electra':
            tokenizer = ElectraTokenizer.from_pretrained(hparams.electra_model_scale,
                                                         do_lower_case=hparams.do_lower_case)

        dev_dataset = NerDataset(dev_examples, tokenizer, label_map, hparams.max_seq_length,
                                 ner_label_map=ner_label_map, pos_label_map=pos_label_map)
        test_dataset = NerDataset(test_examples, tokenizer, label_map, hparams.max_seq_length,
                                  ner_label_map=ner_label_map, pos_label_map=pos_label_map)

        dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                         batch_size=hparams.batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=NerDataset.pad)

        test_dataloader = data.DataLoader(dataset=test_dataset,
                                          batch_size=hparams.batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=NerDataset.pad)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        if hparams.language_model == 'bert':
            lm_model = BertModel.from_pretrained(hparams.bert_model_scale)
        elif hparams.language_model == 'electra':
            lm_model = ElectraModel.from_pretrained(hparams.electra_model_scale)

        model = BERT_CRF_NER(lm_model, start_label_id, stop_label_id, len(label_list),
                             hparams.batch_size,
                             device,
                             num_ner_labels=len(ner_label_list),
                             ner_start_label_id=start_ner_label_id,
                             ner_stop_label_id=stop_ner_label_id,
                             num_pos_labels=len(pos_label_list),
                             pos_start_label_id=start_pos_label_id,
                             pos_stop_label_id=stop_pos_label_id,
                             language_model=hparams.language_model
                             )
        checkpoint = torch.load(hparams.output_dir + '/ner_bert_crf_checkpoint.pt', map_location='cpu')
        epoch = checkpoint['epoch']
        # valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain  NER_BERT_CRF  model, epoch:', checkpoint['epoch'], 'valid acc:',
            'valid f1:', checkpoint['valid_f1'])

        model.to(device)
        # model_epoch = 14

        dev_precision, dev_recall, dev_f1 = evaluate(model, dev_dataloader, hparams.batch_size, epoch, 'DEV')
        test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, hparams.batch_size, epoch, 'TEST')


        dev_write_file = open('/'.join(hparams.output_dir.split('/')[:-1]) + '/dev_results.txt', 'a+')
        test_write_file = open('/'.join(hparams.output_dir.split('/')[:-1]) + '/test_results.txt', 'a+')

        dev_write_file.write('P:%s R:%s F1:%s \n' % (dev_precision, dev_recall, dev_f1))
        test_write_file.write('P:%s R:%s F1:%s \n' % (test_precision, test_recall, test_f1))

        dev_write_file.close()
        test_write_file.close()