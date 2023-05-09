from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import glob






class Encoder(nn.Module):
    def __init__(self,ch_size,pdrop,len_emb):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.drop=torch.nn.Dropout(p=pdrop)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(ch_size, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(
            nn.Linear(8192, 4000),
            torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(
            nn.Linear(4000, 400),
            torch.nn.ReLU())
        self.fc3 = torch.nn.Sequential(
            nn.Linear(400, len_emb),
            torch.nn.ReLU())
    def forward(self, x):
        #print('x0',x.size())
        out = self.conv1(x)
        #print(out.size())
        out = self.maxpool(out)
        out = self.drop(out)
        #print('out00',out.size())
        out = self.conv2(out)
        #print('out00',out.size())
        out = self.maxpool(out)
        out = self.drop(out)
        #print('out0',out.size())
        out = self.conv3(out)
        #print('out1',out.size(),torch.flatten(out,1).size())
        z = self.fc1(torch.flatten(out,1))
        z = self.fc2(z)
        z = self.fc3(z)
        return z

    
class Decoder(nn.Module):
    def __init__(self,ch_size,pdrop,len_emb):
        super().__init__()
        
        self.unflatten = nn.Unflatten(1, (32, 16, 16))
        self.drop=torch.nn.Dropout(p=pdrop)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, ch_size, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        self.fc3 = torch.nn.Sequential(
            nn.Linear(4000, 8192),
            torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(
            nn.Linear(400, 4000),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(
            nn.Linear(len_emb, 400),
            torch.nn.ReLU())
    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.unflatten(z)
        #print('z_dec',z.size())
        out = self.conv1(z) 
        out = torch.nn.functional.interpolate(out, scale_factor=50/16, mode='nearest')
        #print('out_dec',out.size())
        out = self.conv2(out)
        out = torch.nn.functional.interpolate(out, scale_factor=3, mode='nearest')
        out = self.conv3(out)
        #print('out_f',out.size())
        return out
    



class AutoEnc(nn.Module):
    def __init__(self,ch_size,pdrop,len_emb):
        super().__init__()
        self.encoder= Encoder(ch_size,pdrop,len_emb)
        self.decoder = Decoder(ch_size,pdrop,len_emb)


    def forward(self,x):
        z=self.encoder(x)
        #print('autoEnc',z.size())
        x=self.decoder(z)

        return x 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs_path = "emb_train/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.file_paths = []
        for img_path in glob.glob(self.imgs_path + "/*.npy"):
                self.file_paths.append(img_path)
        
       
    
    def __getitem__(self, index):
        # Load the image
        image = torch.from_numpy(np.load(self.file_paths[index]))
        
        # Normalize the image
        image = image / 255.0
        
        # Add channel dimension
        image = image.unsqueeze(0)
        
        #print('image',image.size(),torch.max(image))

        return image, image
    
    def __len__(self):
        return len(self.file_paths)
    



batch_size = 32
learning_rate = 0.001
num_epochs = 10

dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


criterion = torch.nn.MSELoss()

model=AutoEnc(1,0.05,64)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass
        #print('data',data.size())
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss.item()))

# Save the encoder weights
torch.save(model.encoder.state_dict(), 'encoder_weights.pt')
