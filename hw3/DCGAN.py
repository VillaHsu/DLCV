from __future__ import print_function
#%matplotlib inline

from utils import *
from models import *
from torch.nn import functional as F
from torch.autograd import Variable
import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import time
import argparse
import skimage.io
import skimage
import pickle
import sys

# Setup Random Seed
manualSeed = 5555
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
latent_dim = 100
BATCH_SIZE = 128

def train(n_epochs, train_loader):
    rand_inputs = Variable(torch.randn(32,latent_dim, 1, 1), volatile=True)
    G = Generator()
    D = Discriminator()
    if torch.cuda.is_available():
        rand_inputs = rand_inputs.cuda()
        G.cuda()
        D.cuda()

    # setup optimizer
    optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    D_loss_list = []
    G_loss_list = []
    D_real_acc_list = []
    D_fake_acc_list = []

    print("START training...")

    for epoch in range(n_epochs):
        start = time.time()
        D_total_loss = 0.0
        G_total_loss = 0.0
        Real_total_acc = 0.0
        Fake_total_acc = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = len(data)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
            data = to_var(data)
            real_labels = to_var(real_labels)
            fake_labels = to_var(fake_labels)
            
            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #
            
            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            D.zero_grad()
            outputs = D(data)
            D_loss_real = criterion(outputs, real_labels)
            D_accu_real = np.mean((outputs > 0.5).cpu().data.numpy())

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(batch_size, latent_dim, 1, 1)
            z = to_var(z)
            fake_images = G(z)
            outputs = D(fake_images.detach())
            D_loss_fake = criterion(outputs, fake_labels)
            D_accu_fake = np.mean((outputs < 0.5).cpu().data.numpy())

            # Backprop and optimize
            D_loss = D_loss_real + D_loss_fake
            D_total_loss += D_loss.data[0]
            D_loss.backward()
            optimizerD.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #
            
            # Compute loss with fake images
            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            G.zero_grad()
            z = torch.randn(batch_size, latent_dim, 1, 1)
            z = to_var(z)
            fake_images = G(z)
            outputs = D(fake_images)
            G_loss = criterion(outputs, real_labels)
            G_total_loss += G_loss.data[0]
            G_loss.backward()
            optimizerG.step()

            Real_total_acc += D_accu_real
            Fake_total_acc += D_accu_fake
            if batch_idx % 5 == 0:      
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| D_Loss: {:.6f} , G_loss: {:.6f}| Real_Acc: {:.6f} , Fake_Acc: {:.6f}| Time: {}  '.format(
                    epoch+1, (batch_idx+1) * len(data), len(train_loader.dataset),
                    100. * batch_idx * len(data)/ len(train_loader.dataset),
                    D_loss.data[0] / len(data), G_loss.data[0] / len(data),
                    D_accu_real, D_accu_fake,
                    timeSince(start, (batch_idx+1)*len(data)/ len(train_loader.dataset))),end='')
    
        print('\n====> Epoch: {} \nD_loss: {:.6f} | Real_Acc: {:.6f} \nG_loss: {:.6f} | Fake_Acc: {:.6f}'.format(
            epoch+1, D_total_loss/len(train_loader.dataset), Real_total_acc/len(train_loader),
            G_total_loss/len(train_loader.dataset), Fake_total_acc/len(train_loader)))
        print('-'*88)

        D_loss_list.append(D_total_loss/len(train_loader.dataset))
        G_loss_list.append(G_total_loss/len(train_loader.dataset))
        D_real_acc_list.append(Real_total_acc/len(train_loader))
        D_fake_acc_list.append(Fake_total_acc/len(train_loader))

        G.eval()
        rand_outputs = G(rand_inputs)
        G.train()
        torchvision.utils.save_image(rand_outputs.cpu().data,
                                './output_imgs/gan/fig1_2_%03d.jpg' %(epoch+1), nrow=8)
        
        torch.save(G.state_dict(), './saves/save_models/Generator_%03d.pth'%(epoch+1))
        torch.save(D.state_dict(), './saves/save_models/Discriminator_%03d.pth'%(epoch+1))

    with open('./saves/gan/D_loss.pkl', 'wb') as fp:
        pickle.dump(D_loss_list, fp)
    with open('./saves/gan/G_loss.pkl', 'wb') as fp:
        pickle.dump(G_loss_list, fp)
    with open('./saves/gan/D_real_acc.pkl', 'wb') as fp:
        pickle.dump(D_real_acc_list, fp)
    with open('./saves/gan/D_fake_acc.pkl', 'wb') as fp:
        pickle.dump(D_fake_acc_list, fp)


def main(args): 
    TRAIN_DIR = os.path.join(args.train_path, 'train')
    TRAIN_CSVDIR = os.path.join(args.train_path, 'train.csv')

    train_data = CelebADataset('GAN',TRAIN_DIR, TRAIN_CSVDIR)

    print("Read Data Done !!!")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    print("Enter Train")
    train(150, train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN Example')
    parser.add_argument('--train_path', help='training data directory', type=str)
    args = parser.parse_args()
    main(args)    
'''
# Input parameters
dataroot = "hw3_data/face"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = 1

# Load dataset and prepare dataloader 
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Setup GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create the generator
netG = Generator(ngpu).to(device)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)



# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))





# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training ...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists(os.path.join('checkpoints','{}_{}'.format('Generator', manualSeed))):
    os.makedirs(os.path.join('checkpoints', '{}_{}'.format('Generator', manualSeed)))
if not os.path.exists(os.path.join('checkpoints','{}_{}'.format('Discriminator', manualSeed))):
    os.makedirs(os.path.join('checkpoints', '{}_{}'.format('Discriminator', manualSeed)))

save_model(netG, os.path.join('checkpoints', 
                                       '{}_{}'.format('Generator', manualSeed),
                                       'model_{}_pth.tar'.format('Generator')))
save_model(netD, os.path.join('checkpoints', 
                                       '{}_{}'.format('Discriminator', manualSeed),
                                       'model_{}_pth.tar'.format('Discriminator')))
'''