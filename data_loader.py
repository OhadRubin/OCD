import torch
import torch.nn as nn
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
import transformers
from transformers import AutoModel,AutoTokenizer,AutoModelForSequenceClassification
import datasets
import torch.nn.functional as F

name = "w11wo/javanese-bert-small-imdb-classifier"
class MyBert(nn.Module):
    # "weight_name": "model.bert.pooler.dense",
    def __init__(self):
        super(MyBert, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        
    def forward(self, x):
        output = self.model.bert(input_ids=x,output_hidden_states=True)
        pooler_input = output.hidden_states[-1][:,0]
        pooler_input = pooler_input.view(-1, pooler_input.shape[-1])
        latent_in  = deepcopy(pooler_input.detach())
        pooled_output = self.model.dropout(output.pooler_output)
        logits = self.model.classifier(pooled_output)
        pooled_output = pooled_output.view(-1, pooled_output.shape[-1])
        latent = deepcopy(pooled_output.detach())
        # output = self.fc2(output)
        return logits,(latent,latent_in)
    
class RankOne(nn.Module):
    def __init__(self):
        super(RankOne, self).__init__()
        self.dense = nn.Linear(4,768, bias=False)
        # self.dense.weight.data = 0*self.dense.weight.data
    def forward(self,x,dense):
        # params = self.weight
        params = self.dense.weight.T


        alpha,beta,lamda,eta = params
        ab = alpha[None,:]*beta[:,None]
        le = lamda[None,:]*eta[:,None]
        rankone_weight =  -(dense.weight@ab)+le
        y = dense(x)+(x@rankone_weight)
        # print(y.requires_grad)
        return F.tanh(y)

class MyAdaptedBert(nn.Module):
    def __init__(self):
        super(MyAdaptedBert, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        self.rankone = RankOne()
        
    def forward(self, x):
        # with torch.no_grad():
        output = self.model.bert(input_ids=x,output_hidden_states=True)
        dim = self.model.config.hidden_size
        pooler_input = output.hidden_states[-1][:,0].view(-1, dim)
        pooled_output = self.rankone(pooler_input,dense=self.model.bert.pooler.dense)
        latent_in  = deepcopy(pooler_input.detach())
        # pooled_output =self.model.bert.pooler(output.hidden_states[-1])
        # print(pooled_output)
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        pooled_output = pooled_output.view(-1, dim)
        latent = deepcopy(pooled_output.detach())
        
        return logits,(latent,latent_in)

def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':
        
        data =  np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] =get_minibatches
        batch['chunksize'] =chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [],[]
        for img,tfrom in zip(images,tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
                            [
                            transforms.ToTensor()
                            ])
        train_dataset = mnist.MNIST(
                "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
                "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [],[]
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype.startswith('bert'):
        
        # model = AutoModel.from_pretrained(name)
        if args.datatype=="bert":
            model = MyBert()
        else:
            model = MyAdaptedBert()
        dataset = datasets.load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained(name)
        # train_dataset = mnist.MNIST(
        #         "\data\mnist", train=True, download=True, transform=ToTensor())
        # test_dataset = mnist.MNIST(
        #         "\data\mnist", train=False, download=True, transform=ToTensor())
        train_dataset = dataset['train']
        
        test_dataset = dataset['test']
        def add_input_ids(batch):
            batch['input_ids'] = tokenizer(batch['text'], truncation=True,return_tensors="np")['input_ids']
            del batch['text']
            return batch
        train_dataset = train_dataset.map(add_input_ids, batched=True)
        test_dataset = test_dataset.map(add_input_ids, batched=True)
        
        # train_dataset = train_dataset.select(range(100))
        def map_func(batch):
            res = {}
            res['input'] = batch['input_ids']
            # [None,:]
            res['output'] = batch['label']
            # [...,None]
            return res
        train_dataset = train_dataset.map(map_func,batched=True)
        test_dataset = test_dataset.map(map_func,batched=True)
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")
        train_dataset = train_dataset.shuffle(42)
        test_dataset = test_dataset.shuffle(42)
        test_dataset = test_dataset.select(range(100,20000))
        # train_dataset = train_dataset.select(range(5000))
        small_test_dataset = test_dataset.select(range(100))
        train_ds = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_ds = DataLoader(test_dataset, batch_size=1)
        small_ds = DataLoader(small_test_dataset, batch_size=1)
        # test_loader = DataLoader(test_dataset, batch_size=1)
        
        # train_ds, test_ds = [],[]
        # for idx, data in enumerate(train_loader):
            
            
        #     batch = {'input':torch.tensor(data['input_ids']),'output':data['label'].unsqueeze(1)}
        #     train_ds.append(batch)
        # for idx, data in enumerate(test_loader):
            
            
        #     batch = {'input':torch.tensor(data['input_ids']),'output':data['label'].unsqueeze(1)}
        #     test_ds.append(batch)

    else:
        "implement on your own"
        pass
    return train_ds,test_ds,small_ds,model
