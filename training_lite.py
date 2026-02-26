import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from tqdm import tqdm
import math

from consegnaGENAI.consegna_GenAI_DiffusionModel.architecture import ConditionalUNet

class DDPMScheduler:
    '''
    classe che implementa il DDPM Noise Scheduler
    '''
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calcoli per diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calcoli per posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = ( #
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def add_noise(self, x_start, t, noise):
        """
        forward diffusion: q(x_t | x_0) (fa il diffusion kernel)
        """
        # serve la reshape perché x_start è (batch_size, C, H, W), con reshape aggiungiamo le dimensioni a alpha_t
        # che invece è solo (batch_size,)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise # diffusion kernel

    def sample_timesteps(self, n, device):
        """
        sample uniforme di n random timesteps
        """
        return torch.randint(0, self.num_timesteps, (n,), device=device)

    def to(self, device):
        """
        Sposta tutti i tensori sul device specificato
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        return self

    def get_state_dict(self):
        """
        Restituisce lo state dict per salvare lo scheduler
        """
        return {
            'num_timesteps': self.num_timesteps,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'alphas_cumprod_prev': self.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'posterior_variance': self.posterior_variance,
            'sqrt_recip_alphas': self.sqrt_recip_alphas,
            'sqrt_recipm1_alphas_cumprod': self.sqrt_recipm1_alphas_cumprod,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """
        Carica lo scheduler da uno state dict
        """
        scheduler = cls(num_timesteps=state_dict['num_timesteps'])
        scheduler.betas = state_dict['betas']
        scheduler.alphas = state_dict['alphas']
        scheduler.alphas_cumprod = state_dict['alphas_cumprod']
        scheduler.alphas_cumprod_prev = state_dict['alphas_cumprod_prev']
        scheduler.sqrt_alphas_cumprod = state_dict['sqrt_alphas_cumprod']
        scheduler.sqrt_one_minus_alphas_cumprod = state_dict['sqrt_one_minus_alphas_cumprod']
        scheduler.posterior_variance = state_dict['posterior_variance']
        scheduler.sqrt_recip_alphas = state_dict['sqrt_recip_alphas']
        scheduler.sqrt_recipm1_alphas_cumprod = state_dict['sqrt_recipm1_alphas_cumprod']
        return scheduler

# Generazione Immagini
@torch.no_grad()
def sample_images(model, scheduler, conditions, img_size=64, device='cuda'):
    """
    Genera immagini usando DDPM sampling
    
    Args:
        model: Il modello UNet
        scheduler: DDPMScheduler
        conditions: Tensor (N, 3) con le condizioni
        img_size: Dimensione immagine
        device: Device
    """
    model.eval()
    n = conditions.shape[0] # numero di campioni da creare
    
    # Rumore di partenza
    z_t = torch.randn(n, 3, img_size, img_size, device=device)
    
    # Reverse diffusion process
    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long) # otteniamo tanti t quanti batch
        
        # Predict noise
        predicted_noise = model(z_t, t_batch, conditions)
        
        # Calcolo alpha, alpha_hat, beta
        alpha = scheduler.alphas[t]
        alpha_hat = scheduler.alphas_cumprod[t]
        beta = scheduler.betas[t]
        
        # Formula DDPM
        if t > 0:
            noise = torch.randn_like(z_t)
        else:
            noise = torch.zeros_like(z_t)
        
        # z_{t-1} = 1/sqrt(alpha) * (z_t - beta/sqrt(1-alpha_hat) * predicted_noise) + sqrt(beta) * noise
        mu_zt_w_t_c = (1 / torch.sqrt(alpha)) * (z_t - ((beta) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        z_t = mu_zt_w_t_c + torch.sqrt(beta) * noise
    
    model.train()
    return z_t # x


# Training Loop
def train(
    model,
    dataloader,
    scheduler,
    optimizer,
    device,
    num_epochs=100,
    save_dir='checkpoints',
    samples_dir='samples',
    start_epoch=0,
    ema_decay=0.995
):
    """
    Training loop principale
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # EMA (Exponential Moving Average) per stabilità
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay) # θ_EMA(t) = α * θ_EMA(t-1) + (1 - α) * θ_model(t) dove α è l'ema_decay

    )
    
    criterion = nn.MSELoss()
    
    # Condizioni per sampling (tutte le 8 combinazioni)
    sample_conditions = torch.tensor([
        [0, 0, 0],  # Female, Not Smiling, Not Young
        [0, 0, 1],  # Female, Not Smiling, Young
        [0, 1, 0],  # Female, Smiling, Not Young
        [0, 1, 1],  # Female, Smiling, Young
        [1, 0, 0],  # Male, Not Smiling, Not Young
        [1, 0, 1],  # Male, Not Smiling, Young
        [1, 1, 0],  # Male, Smiling, Not Young
        [1, 1, 1],  # Male, Smiling, Young
    ], dtype=torch.float32, device=device)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, conditions) in enumerate(progress_bar):
            images = images.to(device)
            conditions = conditions.to(device)
            
            batch_size = images.shape[0]
            
            # Sample timesteps
            t = scheduler.sample_timesteps(batch_size, device)
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # aggiunta noise alle images
            noisy_images = scheduler.add_noise(images, t, noise) # diffusion kernel
            
            # Predict noise
            predicted_noise = model(noisy_images, t, conditions)
            
            # Calcolo loss
            loss = criterion(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # serve a evitare gradienti troppo grandi
            
            optimizer.step()
            
            # Update EMA
            ema_model.update_parameters(model)
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.6f}')
        
        # Salva checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.get_state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        torch.save(checkpoint, os.path.join(save_dir, 'latest.pt'))
        
        # Generate samples con EMA model
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('Generazione sample')
            ema_model.eval()
            samples = sample_images(ema_model.module, scheduler, sample_conditions, device=device)
            ema_model.train()
            
            # Denormalizzazione [-1, 1] -> [0, 1]
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            
            # Creazione griglia
            grid = make_grid(samples, nrow=4, padding=2)
            save_image(grid, os.path.join(samples_dir, f'samples_epoch_{epoch+1}.png'))


# Data Loading
def get_dataloader(batch_size=32, num_workers=4, data_root='/home/pfoggia/GenerativeAI/CELEBA'):
    """
    Carica CelebA dataset con le trasformazioni appropriate
    """
    # Attributi: Male (#20), Smiling (#31), Young (#39)
    attr_indices = [20, 31, 39]  # Male, Smiling, Young
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    dataset = datasets.CelebA(
        root=data_root,
        split='train',
        transform=transform,
        download=True
    )
    
    # Custom collate function per estrarre solo gli attributi necessari
    def collate_fn(batch):
        '''
        serve perché altrimenti si prendono tutti e 40 attributi di CelebA nel montaggio dei batch
        '''
        images = torch.stack([item[0] for item in batch])
        # Estrai solo gli attributi che ci interessano
        attrs = torch.stack([item[1][attr_indices].float() for item in batch])
        return images, attrs
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def main():
    # Iperparametri
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-4
    NUM_TIMESTEPS = 1000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {DEVICE}')
        
    # modello
    model = ConditionalUNet(img_ch=3, cond_dim=3, base_ch=128).to(DEVICE)
    
    # scheduler
    scheduler = DDPMScheduler(num_timesteps=NUM_TIMESTEPS).to(DEVICE)
    
    # optimizer AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # caricamento checkpoint
    start_epoch = 0
    checkpoint_path = 'checkpoints/latest.pt'
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = DDPMScheduler.from_state_dict(checkpoint['scheduler_state_dict']).to(DEVICE)
        start_epoch = checkpoint['epoch']
        print(f'Resuming from epoch {start_epoch}')
    
    # load data
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    print(f'Dataset size: {len(dataloader.dataset)}')
    
    # Train
    train(
        model=model,
        dataloader=dataloader,
        scheduler=scheduler,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch
    )


if __name__ == '__main__':
    main()