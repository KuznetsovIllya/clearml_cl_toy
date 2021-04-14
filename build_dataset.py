import random
import os
import shutil
import glob
import argparse
import pdb

image_types = ".jpg"

def parse_args():
    ap = argparse.ArgumentParser("Dataset builder")
    ap.add_argument("--data_path", help="path to folder with dataset")
    ap.add_argument("--test_split", help="train data percentage")
    ap.add_argument("--eval_split", help="eval data percentage")
    
    ap.add_argument("--output_dir", default="output", help="path to store splitted dataset")

    return vars(ap.parse_args())

def list_images(path, contains=None):
    return list_files(path, valid_ext=image_types, contains=contains)

def list_files(path, valid_ext=None, contains=None):
    images_list = []
    for (root_dir, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if contains is not None and contains not in filename:
                continue

            # pdb.set_trace()
            if valid_ext is not None and filename.endswith(valid_ext):
                image_path = os.path.join(root_dir, filename)
                yield image_path

def split_train_eval(images_list, test_perc, eval_perc):
    sep = int(len(images_list) * test_perc)
    train_images = images_list[sep:]
    test_images = images_list[:sep]

    eval_sep = int(len(train_images) * eval_perc)
    eval_images = train_images[:eval_sep]
    train_images = train_images[eval_sep:]

    return [
        ("train", train_images),
        ("eval", eval_images),
        ("test", test_images)
    ]

def copy_datasets(datasets, dst_path):
    for dataset_type, paths in datasets:
        print("Building %s dataset" % dataset_type)

        if not os.path.exists(os.path.join(dst_path, dataset_type)):
            os.makedirs(os.path.join(dst_path, dataset_type))
        for path in paths:
            filename = path.split(os.path.sep)[-1]
            label = path.split(os.path.sep)[-2]

            labelPath = os.path.join(dst_path, dataset_type, label)
            if not os.path.exists(labelPath):
                print("Creating %s directory" % labelPath)
                os.makedirs(labelPath)
            
            dst = os.path.join(labelPath, filename)
            shutil.copy2(path, dst)



def main(args):

    dataset_path = args["data_path"]

    images_list = list(list_images(dataset_path))

    random.seed(42)
    random.shuffle(images_list)

    datasets = split_train_eval(images_list, float(args["test_split"]), float(args["eval_split"]))
    copy_datasets(datasets, args["output_dir"])


if __name__ == "__main__":
    args = parse_args()

    main(args)
