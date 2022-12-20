import random

random.seed(123)    

def split(origin_txt, train_txt, val_txt, ratio=0.9):
    files = [line.strip() for line in open(origin_txt, 'r')]

    train_files = random.choices(files, k=int(len(files) * ratio))
    with open(train_txt, 'w') as f:
        flag = False
        for file in train_files:
            if flag:
                f.write('\n' + file)
            else:
                flag = True
    
    val_files = [file for file in files if file not in train_files]
    with open(val_txt, 'w') as f:
        flag = False
        for file in val_files:
            if flag:
                f.write('\n' + file)
            else:
                flag = True


if __name__ == '__main__':
    split(
        './dataset/train_origin.txt', 
        './dataset/train.txt', 
        './dataset/val.txt'
    )
