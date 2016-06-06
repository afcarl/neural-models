import numpy as np

from os.path import isfile, join, splitext, basename
import h5py

from sklearn.cross_validation import StratifiedShuffleSplit

def folder2labels(path):
    #TODO
    pass



def preprocessing(data, train_set_h5, test_set_h5=None):
    if test_set == None:
        data_train = data
    else:
        data_train, data_test = split_traintest(data)

    save_img_h5(data_train, train_set_h5)

    if test_set_h5 != None:
        save_img_h5(data_test, test_set_h5)



def save_img_h5(data, h5_file, cache_size=1600):
    f = h5py.File(h5_file, "w")
    files = [name for (name,l) in data]
    labels = np.array([l for (name,l) in data])
    n_samples = len(data)

    files_dset = f.create_dataset("files", data=files)
    
    lab_dset = f.create_dataset("labels", data=labels,compression="gzip")
    
    img_dset = f.create_dataset("imgs", (n_samples,3,256,256),
                                dtype='float32',
                                compression="gzip")
    n_step = len(data)/cache_size
    if n_step % cache_size != 0:
        n_step += 1
    for j in range(n_step):
        print(str(j)+"/"+str(n_step))
        X = preprocess_image_batch(files[j*cache_size:(j+1)*cache_size])
        img_dset[j*cache_size:(j+1)*cache_size] = X
    f.close()



def main(data_file, h5path, test_set=None):
    data = []
    with open(data_file, "r") as f:
        for l in f:
            path, label = l.split(",")
            label = int(label[:-1])
            data.append((path, label))

    preprocessing(data, train_set, test_set_h5 = test_set)
    
    

if __name__ == "__main__":
    parser = argparser.ArgumentParser()
    parser.add_argument("DATA", help = "CSV file containing images and labels")
    parser.add_argument("H5PATH", help= "Path to the preprocessed data")
    parser.add_argument("-ts", "--testset", default=None,
                        help = ("If path is given, it splits the data into "
                                "train and test set")
                        )
    args = parser.arg_parser()
    main(args.DATA, args.H5PATH, test_set = args.testset)
    
