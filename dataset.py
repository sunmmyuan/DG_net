from torch.utils.data import Dataset
import torch
from generate_data import Generate_discon_data, BURGER
import json
import os
import numpy as np
# build a dataset to generate data and be exacted by dataloader


class getdataset(Dataset):
    def __init__(self, data_generator) -> None:
        super().__init__()
        self.data_generator = data_generator
        self.labelfile = self.data_exist()

    def data_exist(self):
        if os.path.exists(os.path.join(self.data_generator.save_path, f'{self.data_generator.status}.json')):
            with open(os.path.join(self.data_generator.save_path, f'{self.data_generator.status}.json'), 'r', encoding='utf-8') as f:
                temp=json.load(f)
                temp=[{'data':np.array(i['data']), 'label':i['label']} for i in temp]
                return temp
        else:
            return self.data_generator.get_data()

    def __getitem__(self, index):
        x = torch.Tensor(self.labelfile[index]['data'])
        y = self.labelfile[index]['label']
        return x, y

    def __len__(self):
        return len(self.labelfile)


def main():
    test=BURGER()
    
    getdata = Generate_discon_data(test.get_disc_point, test.get_R_point)
    getdata.get_data()
    t = getdataset(getdata)

    for i in range(len(t)):
        if t[i][0].shape[0]==5:
            pass
        else:
            print('wrong data', t[i][0])
    # print(t[0][0].shape[0] ==5)


if __name__ == '__main__':
    main()
