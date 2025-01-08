import math
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import Models.Constants as Constants
from data_loader import get_loader 
# from ptp_data_loader import get_loader
from Models.Main import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def cal_loss(pred, target, smoothing=True):
    #''' Calculate cross entropy loss, apply label smoothing if needed. '''
    target = target.contiguous().view(-1)
    
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = target.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # Language Modeling Loss
        LMcriterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            LMcriterion.cuda()
        loss = LMcriterion(pred, target)

    return loss

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    Concepts = json.load(open(args.concept_path, 'r'))
    Model = Encoder2Decoder(args.embed_size, len(vocab), args.hidden_size, args.use_MIA, args.iteration_times)
    if args.pretrained:
        Model.load_state_dict(torch.load(args.pretrained))
        start_epoch = int(args.pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1
    elif args.pretrained_cnn:
        pretrained_dict = torch.load(args.pretrained_cnn)
        model_dict = Model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        Model.load_state_dict(model_dict)
        start_epoch = 1
    else:
        start_epoch = 1
    cnn_subs = list(Model.encoder_image.resnet_conv.children())[args.fine_tune_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]
    '''
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn, 
                                      betas=(args.alpha, args.beta))
    '''
    cnn_optimizer = torch.optim.AdamW(
        cnn_params, 
        lr=args.learning_rate_cnn, 
        betas=(args.alpha, args.beta)
    )
    params = list(Model.encoder_image.affine_a.parameters()) + list(Model.decoder.parameters())
    if args.use_MIA:
        params += list(Model.MIA.parameters())
        
    learning_rate = args.learning_rate
    if torch.cuda.is_available():
        Model.cuda()
    data_loader = get_loader(args.image_dir, args.caption_path, Concepts, vocab,
                             transform, args.train_batch_size, shuffle=True,
                             num_workers=args.num_workers)
    total_step = len(data_loader)
    print(f'len: {total_step}')
    
    optimizer = torch.optim.AdamW(
        params, 
        lr=args.learning_rate, 
        betas=(args.alpha, args.beta)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f'Learning Rate for Epoch {epoch}: {learning_rate:.6f}')

        # optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(args.alpha, args.beta))
        

        print(f'------------------Training for Epoch {epoch}----------------')
        
        for i, (images, captions, lengths, _, _, image_concepts) in enumerate(data_loader):
            images = to_var(images)
            captions = to_var(captions)
            image_concepts = to_var(image_concepts)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
            
            Model.train()
            Model.zero_grad()

            packed_scores = Model(images, captions, image_concepts, lengths, args.basic_model)
            loss = cal_loss(packed_scores[0], targets, smoothing=True)
            loss.backward()
            for p in Model.decoder.parameters():
                p.data.clamp_(-args.clip, args.clip)
            optimizer.step()
            
            if epoch > args.cnn_epoch: 
                cnn_optimizer.step()
                
            if i % args.log_step == 0:
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}')
                
        scheduler.step()

        if args.use_MIA:
            save_path = os.path.join(args.save_dir_path, f"{args.basic_model}-MIA")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = os.path.join(args.save_dir_path, args.basic_model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        torch.save(Model.state_dict(), os.path.join(save_path, f'Model-{epoch}.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument('--save_dir_path', type=str, default='./ckpt1',
                         help='path for saving trained models')
    parser.add_argument('--basic_model', type=str, default='VisualAttention',
                         help='the selected basic model, [VisualAttention, ConceptAttention, VisualCondition, ConceptCondition]')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/images/resized/',
                        help='directory for training images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/karpathy_split_train.json',
                        help='path for train annotation json file')
    parser.add_argument('--concept_path', type=str,
                        default='./data/image_concepts.json',
                        help='path for image concepts json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    
    # Hyper Parameter Setup
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=8,
                        help='start fine-tuning CNN after')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-5,
                         help='learning rate for fine-tuning CNN')
    parser.add_argument('--alpha', type=float, default=0.8,
                         help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                         help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                         help='learning rate for the whole model')
    parser.add_argument('--embed_size', type=int, default=512,
                         help='dimension of word embedding vectors')
    # 512
    parser.add_argument('--hidden_size', type=int, default=512,
                         help='dimension of lstm hidden states')
    parser.add_argument('--use_MIA', type=bool, default=False)
    parser.add_argument('--iteration_times', type=int, default=2, help='the iteration times in mutual iterative attention')
    parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    parser.add_argument('--pretrained_cnn', type=str, default='', help='load pretrained_cnn parameters')
    parser.add_argument('--num_epochs', type=int, default=10)
    # 80
    parser.add_argument('--train_batch_size', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                         help='decay learning rate at every this number')
    
    args = parser.parse_args()
    
    print('------------------------Model and Training Details--------------------------')
    print(args)
    
    main(args)
