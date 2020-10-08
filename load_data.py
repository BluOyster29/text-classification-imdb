import argparse, os
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('training_path', typ=str, help='Path to the folder containing training data')
    p.add_argument('testing_path', type=str, help='Path to the folder containing testing data')
    p.add_argument('n', type=int, help='Total number of samples', default=200)
    p.add_argument('split', type=int, help='Training/Testing Split', default=0.75)
    args = p.parse_args()

    return args

def load_data(path, n):
    
    data = []

    for i in tqdm(os.listdir(path)[:n]):
        with open('{}{}'.format(path,i), 'r') as file:
            data.append(file.read())
            file.close()

    return data

def main(args):

    
    num_training = round(args.n * args.split)
    num_testing = round(args.n - (args.n * args.split))
    training_pos = load_data('{}pos/'.format(args.training_path), num_training)
    training_neg = load_data('{}neg/'.format(args.training_path), num_training)
    testing_pos = load_data('{}pos/'.format(args.testing_path), num_testing)
    testing_neg = load_data('{}neg/'.format(args.testing_path), num_testing)

    return training_pos, training_neg, testing_pos, testing_neg

if __name__ == '__main__':
    args = get_args()
    main(args)