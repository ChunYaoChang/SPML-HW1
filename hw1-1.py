import os
from PIL import Image
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model
import argparse
import random

seed = 902022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, default="all", help="")
parser.add_argument("--output_path", type=str, default="./prediction.json")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--task", type=str, default="")
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
if args.task == 'adv_img':
    start, end = 0, 5
elif args.task == 'adv_img2':
    start, end = 5, 10
else:
    exit(1)

class AdvDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.images = []
        self.labels = []
        for label in range(100):
            for j in range(start, end):
                img = Image.open(os.path.join(data_path, f'{label}_{j}.png'))
                self.images.append(transforms.Compose([transforms.ToTensor()])(img))
                self.labels.append(label)
                
        self.images = torch.stack(self.images)
        # print(self.images.shape)
        tmp_images = self.images.view(self.images.size(0), self.images.size(1), -1)
        self.mean = tmp_images.mean(2).mean(0)
        self.std = tmp_images.std(2).mean(0)
        # print(self.mean, self.std)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        # print(self.images[index])
        image = self.images[index]
        # image = build_transforms
        image = transforms.Compose([transforms.Normalize(self.mean, self.std)])(image)
        label = self.labels[index]
        return image, label


batch_size = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = AdvDataset(data_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
loss_fn = nn.CrossEntropyLoss()

print(f'number of images = {train_set.__len__()}')

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_acc, total_loss = 0, 0
    for image, label in tqdm(loader):
        image, label = image.to(device), label.to(device)
        pred = model(image)
        # print(pred)
        # print(label)
        loss = loss_fn(pred, label)
        total_acc += (pred.argmax(dim=1) == label).sum().item()
        total_loss += loss.item() * image.shape[0]
    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)



class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        for i, m in enumerate(self.models):
            if i == 0:
                o = self.softmax(m(x)).unsqueeze(2)
            else:
                o = torch.cat([o, self.softmax(m(x)).unsqueeze(2)], dim=2)
        return o.sum(dim=2)


model_names = [
    'nin_cifar100',
    'resnet20_cifar100',
    'preresnet20_cifar100',
    'resnext29_32x4d_cifar100',
    'seresnet20_cifar100',
    'sepreresnet20_cifar100',
    'pyramidnet110_a48_cifar100',
    'densenet40_k12_cifar100',
    'xdensenet40_2_k24_bc_cifar100',
    'wrn16_10_cifar100',
    'wrn20_10_1bit_cifar100',
    'ror3_56_cifar100',
    'rir_cifar100',
    'shakeshakeresnet26_2x32d_cifar100',
    'diaresnet20_cifar100',
    'diapreresnet20_cifar100',
]
ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()

train_acc, train_loss = evaluate(ensemble_model, train_loader, loss_fn, device)
print(f'train_acc: {train_acc:.5f}, train_loss: {train_loss:.5f}')

def momenton_iterative_fgsm(model, image, label, loss_fn, epsilon, alpha, num_iter, mu=0.1, rand_start=False):
    adv_image = image.detach().clone()
    g = torch.zeros_like(image) 
    for _ in range(num_iter):
        new_image = adv_image.detach().clone()
        new_image.requires_grad = True
        loss = loss_fn(model(new_image), label)
        loss.backward()
        grad = new_image.grad
        grad_norm = torch.norm(grad.reshape((grad.shape[0], -1)), p=1., dim=1)
        g = mu * g + grad / grad_norm.view((-1, 1, 1, 1))
        new_image = new_image + alpha * g.detach().sign()
        adv_image = torch.max(torch.min(new_image, image + epsilon), image - epsilon)
    return adv_image

def iterative_fgsm(model, image, label, loss_fn, epsilon, alpha, num_iter):
    adv_image = image.detach().clone()
    for _ in range(num_iter):
        new_image = adv_image.detach().clone()
        new_image.requires_grad = True
        loss = loss_fn(model(new_image), label)
        loss.backward()
        new_image = new_image + alpha * new_image.grad.detach().sign()
        adv_image = torch.max(torch.min(new_image, image + epsilon), image - epsilon)
    return adv_image

def fgsm(model, image, label, loss_fn, epsilon):
    adv_image = image.detach().clone()
    new_image = adv_image.detach().clone()
    new_image.requires_grad = True
    loss = loss_fn(model(new_image), label)
    loss.backward()
    new_image = new_image + epsilon * new_image.grad.detach().sign()
    return new_image

mean = torch.tensor(train_set.mean).to(device).view(3, 1, 1)
std = torch.tensor(train_set.std).to(device).view(3, 1, 1)
epsilon = 8/255/std
alpha = epsilon / 10
num_iter = 10

def perform_attack(model, loader, loss_fn, epsilon, alpha, num_iter):
    model.eval()
    adv_acc, adv_loss = 0, 0
    for i, (image, label) in enumerate(tqdm(loader)):
        image, label = image.to(device), label.to(device)
        # adv_image = iterative_fgsm(model, image, label, loss_fn, epsilon, alpha, num_iter)
        # adv_image = fgsm(model, image, label, loss_fn, epsilon)
        adv_image = momenton_iterative_fgsm(model, image, label, loss_fn, epsilon, alpha, num_iter)
        pred = model(adv_image)
        # print(pred)
        # print(label)
        loss = loss_fn(pred, label)
        adv_acc += (pred.argmax(dim=1) == label).sum().item()
        adv_loss += loss.item() * adv_image.shape[0]

        adv_image = ((adv_image * std + mean) * 255).clamp(0, 255).detach().cpu().data.numpy().round().transpose((0, 2, 3, 1))
        adv_examples = adv_image if i == 0 else np.r_[adv_examples, adv_image]
    return adv_examples, adv_acc / len(loader.dataset), adv_loss / len(loader.dataset)

adv_examples, adv_acc, adv_loss = perform_attack(ensemble_model, train_loader, loss_fn, epsilon, alpha, num_iter)
print(f'adv_acc: {adv_acc:.5f}, adv_loss: {adv_loss:.5f}')

def save_pictures(adv_examples, output_path):
    names = []
    for label in range(100):
        for j in range(start, end):
            names.append(f'{label}_{j}.png')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for example, name in zip(adv_examples, names):
        im = Image.fromarray(example.astype(np.uint8))
        im.save(os.path.join(output_path, name))

save_pictures(adv_examples, output_path)
print('Pictures are saved successfully')