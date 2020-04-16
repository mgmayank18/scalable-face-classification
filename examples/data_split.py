import os
import shutil


def split_dataset(data_path):
    # os.listdir(data_path)
    vgg_path = data_path+"/VGGFace2/"
    # os.listdir(vgg_path)
    train_cropped_path = vgg_path+"/train_cropped/"
    split_path = vgg_path+"/train_cropped_split/"
    files = os.listdir(train_cropped_path)
    n_total = len(files)

    if(os.path.exists(split_path)):
        print(f"{split_path} exists, you have probably already done the data split")
        return
    os.mkdir(split_path)
    for i in range(3000):
        shutil.move(train_cropped_path+files[i], split_path)

    # if(split1 == None):
    #     split1 = int(n_total*60/100)
    #     split2 = int(n_total*40/100)
    # split1_path = vgg_path+"/train_cropped_split1/"
    # split2_path = vgg_path+"/train_cropped_split2/"


if __name__ == "__main__":
    split_dataset("../data")
    
