import os
import os.path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from collections import namedtuple
from model_unet import *
import argparse

class CityscapesDataset(Dataset):

    def __init__(self, root, split='train', mode='fine', augment=False):

        self.root = os.path.expanduser(root)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.augment = augment
        self.images = []
        self.targets = []
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 0,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 0,  # building
            12: 0,  # wall
            13: 0,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 0,  # pole
            18: 0,  # polegroup
            19: 0,  # traffic light
            20: 0,  # traffic sign
            21: 0,  # vegetation
            22: 0,  # terrain
            23: 2,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: 3,  # car
            27: 0,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }
        self.mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }

        # Ensure that this matches the above mapping!#!@#!@#
        # For example 4 classes, means we should map to the ids=(0,1,2,3)
        # This is used to specify how many outputs the network should product...
        self.num_classes = 4
        

        # =============================================
        # Check that inputs are valid
        # =============================================
        if mode not in ['fine', 'coarse']:
            raise ValueError('Invalid mode! Please use mode="fine" or mode="coarse"')
        if mode == 'fine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "fine"! Please use split="train", split="test" or split="val"')
        elif mode == 'coarse' and split not in ['train', 'train_extra', 'val']:
            raise ValueError('Invalid split for mode "coarse"! Please use split="train", split="train_extra" or split="val"')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        # =============================================
        # Read in the paths to all images
        # =============================================
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_labelIds.png'.format(self.mode))
                # target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], '{}_color.png'.format(self.mode))
                self.targets.append(os.path.join(target_dir, target_name))

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Mode: {}\n'.format(self.mode)
        fmt_str += '    Augment: {}\n'.format(self.augment)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        
        return fmt_str

    def __len__(self):
        return len(self.images)

    def mask_to_class(self, mask):
        '''
        Given the cityscapes dataset, this maps to a 0..classes numbers.
        This is because we are using a subset of all masks, so we have this "mapping" function.
        This mapping function is used to map all the standard ids into the smaller subset.
        '''
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        '''
        Given the Cityscapes mask file, this converts the ids into rgb colors.
        This is needed as we are interested in a sub-set of labels, thus can't just use the
        standard color output provided by the dataset.
        '''
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def class_to_rgb(self, mask):
        '''
        This function maps the classification index ids into the rgb.
        For example after the argmax from the network, you want to find what class
        a given pixel belongs too. This does that but just changes the color
        so that we can compare it directly to the rgb groundtruth label.
        '''
        mask2class = dict((v, k) for k, v in self.mapping.items())
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in mask2class:
            rgbimg[0][mask == k] = self.mappingrgb[mask2class[k]][0]
            rgbimg[1][mask == k] = self.mappingrgb[mask2class[k]][1]
            rgbimg[2][mask == k] = self.mappingrgb[mask2class[k]][2]
        return rgbimg

    def __getitem__(self, index):

        # first load the RGB image
        image = Image.open(self.images[index]).convert('RGB')

        # next load the target
        target = Image.open(self.targets[index]).convert('L')

        # If augmenting, apply random transforms
        # Else we should just resize the image down to the correct size
        if self.augment:
            # Resize
            image = TF.resize(image, size=(128+10, 256+10), interpolation=Image.BILINEAR)
            target = TF.resize(target, size=(128+10, 256+10), interpolation=Image.NEAREST)
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(128, 256))
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
            # Random vertical flipping
            # (I found this caused issues with the sky=road during prediction)
            # if random.random() > 0.5:
            #    image = TF.vflip(image)
            #    target = TF.vflip(target)
        else:
            # Resize
            image = TF.resize(image, size=(64, 128), interpolation=Image.BILINEAR)
            target = TF.resize(target, size=(64, 128), interpolation=Image.NEAREST)

        # convert to pytorch tensors
        # target = TF.to_tensor(target)
        target = torch.from_numpy(np.array(target, dtype=np.uint8))
        image = TF.to_tensor(image)

        # convert the labels into a mask
        targetrgb = self.mask_to_rgb(target)
        targetmask = self.mask_to_class(target)
        targetmask = targetmask.long()
        targetrgb = targetrgb.long()
         

        # finally return the image pair
        return image, targetmask, targetrgb
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/home/uesr/Desktop/Cityscapes/", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--losstype", type=str, default="segment", help="choose between segment & reconstruction")
args = parser.parse_args()


lr = 0.0002
epochs = 100

img_data = CityscapesDataset(args.datadir, split='train', mode='fine', augment=True)
img_batch = torch.utils.data.DataLoader(img_data, batch_size=16, shuffle=True)
print(img_data)


if args.losstype == "reconstruction":
    recon_loss_func = nn.MSELoss()
    num_classes = 3  # red, blue, green
elif args.losstype == "segment":
    recon_loss_func = nn.CrossEntropyLoss()
    num_classes = img_data.num_classes  # background, road, sky, car
else:
    print("please select a valid loss type (reconstruction or segment)...")
    exit()
    
    
# initiate generator and optimizer
print("creating unet model...")
generator = nn.DataParallel(UnetGenerator(3, img_data.num_classes, 64), device_ids=[i for i in range(1)])
#generator = nn.DataParallel(UnetGenerator(3, img_data.num_classes, 64), device_ids=[i for i in range(args.num_gpu)]).cuda()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

# load pretrained model if it is there
file_model = './unet.pkl'
if os.path.isfile(file_model):
    generator = torch.load(file_model)
    print("    - model restored from file....")
    print("    - filename = %s" % file_model)


# or log file that has the output of our loss
file_loss = open('./unet_loss', 'w')


# make the result directory
if not os.path.exists('./result/'):
    os.makedirs('./result/')
    
# finally!!! the training loop!!!
for epoch in range(epochs):
    
    for idx_batch, (imagergb, labelmask, labelrgb) in enumerate(img_batch):

        # zero the grad of the network before feed-forward
        gen_optimizer.zero_grad()

        # send to the GPU and do a forward pass
        '''
        x = Variable(imagergb).cuda(0)
        y_ = Variable(labelmask).cuda(0)
        '''
        x = Variable(imagergb)
        y_ = Variable(labelmask)
        y = generator.forward(x)

        # we "squeeze" the groundtruth if we are using cross-entropy loss
        # this is because it expects to have a [N, W, H] image where the values
        # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes
        #y_ = torch.squeeze(y_)
        
        if args.losstype == "segment":
            y_ = torch.squeeze(y_)
        

        # finally calculate the loss and back propagate
        loss = recon_loss_func(y, y_)
        file_loss.write(str(loss.item())+"\n")
        loss.backward()
        gen_optimizer.step()

        # every 400 images, save the current images
        # also checkpoint the model to disk
        if idx_batch % 400 == 0:

            # nice debug print of this epoch and its loss
            print("epoch = "+str(epoch)+" | loss = "+str(loss.item()))

            # save the original image and label batches to file
            v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(epoch, idx_batch))
            v_utils.save_image(labelrgb, "./result/label_image_{}_{}.png".format(epoch, idx_batch))
            y_threshed = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
            for idx in range(0, y.size()[0]):
                maxindex = torch.argmax(y[idx], dim=0).cpu().int()
                y_threshed[idx] = img_data.class_to_rgb(maxindex)
            v_utils.save_image(y_threshed, "./result/gen_image_{}_{}.png".format(epoch, idx_batch))
            
            # max over the classes should be the prediction
            # our prediction is [N, classes, W, H]
            # so we max over the second dimension and take the max response
            # if we are doing rgb reconstruction, then just directly save it to file
            '''
            if args.losstype == "segment":
                y_threshed = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
                for idx in range(0, y.size()[0]):
                    maxindex = torch.argmax(y[idx], dim=0).cpu().int()
                    y_threshed[idx] = img_data.class_to_rgb(maxindex)
                v_utils.save_image(y_threshed, "./result/gen_image_{}_{}.png".format(epoch, idx_batch))
            else:
                v_utils.save_image(y.cpu().data, "./result/gen_image_{}_{}.png".format(epoch, idx_batch))
            '''
            # finally checkpoint this file to disk
            torch.save(generator, file_model)
