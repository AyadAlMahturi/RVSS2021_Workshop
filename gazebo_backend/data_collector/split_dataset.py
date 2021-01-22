import sys
import os
from random import sample
from tqdm import tqdm
import numpy as np
import json
import argparse

def main(args):
    dataset_path = os.path.join('dataset', args.dataset_name)
    if not os.path.exists(dataset_path):
        raise Exception('dataset path % s does not exist' % dataset_path)
    all_model_names = next(os.walk(dataset_path))[1]
    counter = 0
    dataset_catalog = {}
    for model in all_model_names:
        dir_temp = os.path.join(dataset_path, model, 'images')
        all_images =  os.listdir(dir_temp)
        for image_name in all_images:
            image_path = os.path.join(model, 'images', image_name)
            label_path = os.path.join(model, 'labels', image_name[:-4]+'_mask.png')
            dataset_catalog[counter] = {'image': image_path, 'label':label_path}
            counter +=1
    # split training and evaluation
    train_ratio = args.training_ratio
    if 0<train_ratio<1:
        sample_size = len(dataset_catalog)
        all_indices = list(np.arange(sample_size))
        train_indices = sample(all_indices, int(train_ratio*sample_size))
        eval_indices = list(set(all_indices) - set(train_indices))
        train_indices = [int(x) for x in train_indices]
        eval_indices = [int(x) for x in eval_indices]
        train_catalog = {key:dataset_catalog[key] for key in train_indices}
        eval_catalog = {key:dataset_catalog[key] for key in eval_indices}
        with open(os.path.join(dataset_path, 'train.json'), 'w') as f_:
            json.dump(train_catalog, f_)
        with open(os.path.join(dataset_path, 'eval.json'), 'w') as f_:
            json.dump(eval_catalog, f_)
    else:
        raise Exception('training ration must be between (0, 1)')
        

if __name__ == '__main__':
    generator_parser = argparse.ArgumentParser(
        description='Split dataset for trainig and evaluation')
    generator_parser.add_argument('--dataset_name', type=str, default='',
                                  help='dataset name')
    generator_parser.add_argument('--training_ratio' ,  type=float, default=0.9,
                                  help='Ratio of trainig over evaluation')
    
    args = generator_parser.parse_args()
    
    main(args)

