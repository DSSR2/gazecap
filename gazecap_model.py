import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from gazecap_data import gazecap_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

class ItrackerImageModel(pl.LightningModule):
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return x

class FaceImageModel(pl.LightningModule):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(13*13*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(pl.LightningModule):
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x



class gazecap_model(pl.LightningModule):
    def __init__(self, data_path, save_path, logger, batch_size=256, workers=10):
        super(gazecap_model, self).__init__()
        
        self.lr = 0.0001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        self.batch_size = batch_size
        self.workers = workers
        self.data_path = data_path
        print("Data path: ", data_path)
        self.save_path = save_path
        
        print('batch_size: ', self.batch_size, '\n',
              'workers: ', self.workers)
        
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*13*13*64, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x
    
    def train_dataloader(self):
        train_dataset = gazecap_data(self.data_path+"/train/", phase='train')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True, pin_memory=True)
        self.logger.log_hyperparams({'Num_train_files': len(train_dataset)})
        return train_loader
    
    def val_dataloader(self):
        dataVal = gazecap_data(self.data_path+"/val/", phase='val')
        val_loader = DataLoader(dataVal, batch_size=self.batch_size, num_workers=self.workers, shuffle=False, pin_memory=True)
        self.logger.log_hyperparams({'Num_val_files': len(dataVal)})
        return val_loader
    
    def training_step(self, batch, batch_idx):
        imFace, imEyeL, imEyeR, faceGrid, y = batch
        y_hat = self(imFace, imEyeL, imEyeR, faceGrid)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.log_metric('train_loss', loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        imFace, imEyeL, imEyeR, faceGrid, y = batch
        y_hat = self(imFace, imEyeL, imEyeR, faceGrid)
        val_loss = F.mse_loss(y_hat, y)
        self.log('val_loss', val_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.log_metric('val_loss', val_loss.item())
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }