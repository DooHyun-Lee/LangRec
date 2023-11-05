import argparse
import os
from torch.utils.data import DataLoader
import torch 
from accelerate import Accelerator 
from data import load_RecListDataset, GeneralDataset
from models.bc_lm import load_bclm, load_evaluator
from tqdm import tqdm
import pickle 
from models.base import to

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/doolee13/LangRec/preprocess', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--max_len', default=1024, type=int)
parser.add_argument('--dataloader_workers', default=1, type=int)
parser.add_argument('--bsize', default=2, type=int)
parser.add_argument('--eval_bsize', default=2, type=int)
parser.add_argument('--gpt2_type', default='gpt2', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--epochs', default=10000000, type=int)
parser.add_argument('--grad_accum_steps', default=128, type=int)
parser.add_argument('--eval_every', default=4096, type=int)
parser.add_argument('--metadata_path', default='/home/doolee13/LangRec/preprocess/meta_data.json', type=str)
parser.add_argument('--checkpoint_dir', default='/home/doolee13/LangRec/outputs', type=str)

args = parser.parse_args()

if __name__ == '__main__': 
    accelerator = Accelerator()
    
    train_path = os.path.join(args.path, 'train_data.json')
    eval_path = os.path.join(args.path, 'valid_data.json')
    path_train = os.path.join('/home/doolee13/LangRec/preprocess', 'RecommendListDataset_train.pkl')
    path_eval = os.path.join('/home/doolee13/LangRec/preprocess', 'RecommendListDataset_validation.pkl')
    if os.path.exists(path_train):
        with open(path_train, 'rb') as f:
            train_raw = pickle.load(f)
    else:
        with open(path_train, 'wb') as f:
            train_raw = load_RecListDataset(train_path, args.max_len) # RecommendListDataset class
            pickle.dump(train_raw, f)
    if os.path.exists(path_eval):
        with open(path_eval, 'rb') as f:
            eval_raw = pickle.load(f)
    else:
        with open(path_eval, 'wb') as f:
            eval_raw = load_RecListDataset(eval_path, args.max_len)
            pickle.dump(eval_raw, f)

    train_dataset = GeneralDataset(train_raw, 'cpu')
    eval_dataset = GeneralDataset(eval_raw, 'cpu')
    train_data_loader_kwargs = {'num_workers' : args.dataloader_workers,
                                'batch_size' : args.bsize,
                                'collate_fn' : train_dataset.collate} 
    eval_data_loader_kwargs = {'num_workers' : args.dataloader_workers,
                                'batch_size' : args.eval_bsize,
                                'collate_fn' : eval_dataset.collate} 
    train_data_loader_kwargs['shuffle'] = True
    eval_data_loader_kwargs['shuffle'] = False
    data_loader = DataLoader(train_dataset, **train_data_loader_kwargs)
    eval_data_loader = DataLoader(eval_dataset, **eval_data_loader_kwargs)
    model = load_bclm(args, train_raw) # moved on to device inside
    model.train()
    evaluator = load_evaluator(args, model)
    params = model.parameters()
    optim = torch.optim.AdamW(params, lr = args.lr, weight_decay=args.weight_decay)
    model, optim, data_loader, eval_data_loader = accelerator.prepare(model, optim, data_loader, eval_data_loader)
    step = 0
    loss_log = 0 
    max_loss_log = float('inf')
    for epoch in tqdm(range(args.epochs)):
        for items in tqdm(data_loader):
            items = to(items, device=args.device)
            loss, logs, _ = accelerator.unwrap_model(model).get_loss(items)
            accelerator.backward(loss/args.grad_accum_steps)
            loss_log += (loss.item() / args.grad_accum_steps)
            if (step+1) % args.grad_accum_steps == 0:
                optim.step()
                optim.zero_grad()
                print(f'current loss at {step} : {loss_log}')
                loss_log = 0
            if (step+1) % args.eval_every == 0 :
                model.eval()
                with torch.no_grad():
                    eval_loss_log = 0
                    for i, eval_items in enumerate(eval_data_loader):
                        eval_items = to(eval_items, device=args.device)
                        loss, logs, _ = accelerator.unwrap_model(model).get_loss(eval_items)
                        eval_loss_log += loss.item()
                        # only printout one example sentence
                        if i == len(eval_data_loader) -1:
                            test_act = evaluator.evaluate(eval_items)
                            print(f'sample sentence at {step} : {test_act[0]}')
                    eval_loss_log /= len(eval_data_loader)
                    print(f'evaluation loss at {step} : {eval_loss_log}')
                    if max_loss_log > eval_loss_log:
                        max_loss_log = eval_loss_log
                        if not os.path.exists(args.checkpoint_dir):
                            os.makedirs(args.checkpoint_dir)
                        torch.save(accelerator.unwrap_model(model).state_dict(), 
                                   os.path.join(args.checkpoint_dir, 'model.pkl'))
                        torch.save(optim.state_dict(),
                                   os.path.join(args.checkpoint_dir, 'optim.pkl'))
                        print(f'new checkpoint saved at timestep :{step}')

                model.train()
            step += 1