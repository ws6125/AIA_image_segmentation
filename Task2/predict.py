import os
import numpy as np
import pandas as pd
import torch
import cv2
import unet
import attunet
import dataset
import utils

from torch.utils.data import DataLoader

path = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = 'best_model.pt'

# hyper parameters
batch_size = 1
threshold = 0

def mask2rle(pred):
    img = pred

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def test():
    # model
    # model = unet.UNet(n_channels = 1, n_classes = 1)
    model = attunet.AttU_Net(n_channels = 1, n_classes = 1)
    model.to(device = device)
    model.load_state_dict(torch.load(best_model, map_location = device))
    model.eval()

    # dataset
    test_dataset = dataset.TestDataset(output_ori_image = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)
    idx = 0
    output_list = []
    for data, img_path in test_loader:
        for i in range(0, len(data)):
            img = data[i]
            file_name = '-'.join([str(img_path[i].split('/')[idx]) for idx in [-4,-3,-1]])
            output_path = (img_path[i]).replace('testset', 'Pred').replace('.dcm', '_res.png')

            # create folder if not exists
            dir = '/'.join((output_path.split('/'))[: -1]) + '/'
            utils.checkFolder(dir)

            print(f'[ Output | {output_path} ]')

            # predict
            img = img.reshape(1, 1, img.shape[1], img.shape[2])
            img = img.to(device = device, dtype = torch.float32)
            pred = model(img)
            pred = np.array(pred.data.cpu()[0])[0]

            # set mask
            pred[threshold <= pred] = 255
            pred[threshold > pred] = 0

            # resize
            # weight, height = test_dataset.getOriImgSize(idx)
            # pred = cv2.resize(pred, (weight, height))

            # write predict image
            cv2.imwrite(output_path, pred)
            # cv2.imshow('Pred', pred)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            output_list.append([file_name, mask2rle(pred)])

            idx += 1

    pd.DataFrame(output_list, columns = ['filename', 'rle']).to_csv('ST_submission.csv', index = False)

if '__main__' == __name__:
    test()
