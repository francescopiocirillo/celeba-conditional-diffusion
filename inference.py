import torch
from torchvision.utils import save_image, make_grid
import os

from consegnaGENAI.consegna_GenAI_DiffusionModel.architecture import ConditionalUNet
from consegnaGENAI.consegna_GenAI_DiffusionModel.training_lite import DDPMScheduler, sample_images


# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './latest.pt'
OUT_DIR = 'generated'
IMG_SIZE = 64
N_SAMPLES = 8

# CONDIZIONAMENTO
CONDITION_VECTOR = [1, 1, 0]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Modello
    model = ConditionalUNet(img_ch=3, cond_dim=3, base_ch=128).to(DEVICE)

    # Carica checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Scheduler
    scheduler = DDPMScheduler.from_state_dict(
        checkpoint['scheduler_state_dict']
    ).to(DEVICE)

    # Condizionamento
    condition = torch.tensor(
        CONDITION_VECTOR,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0).repeat(N_SAMPLES, 1) # (replicato 8 volte)

    # Sampling
    with torch.no_grad():
        samples = sample_images(
            model=model,
            scheduler=scheduler,
            conditions=condition,
            img_size=IMG_SIZE,
            device=DEVICE
        )

    # Denormalizzazione [-1,1] → [0,1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Salva griglia
    grid = make_grid(samples, nrow=4, padding=2)
    grid_path = os.path.join(
        OUT_DIR,
        f'samples_same_cond_{CONDITION_VECTOR}.png'
    )
    save_image(grid, grid_path)

    print(f'{N_SAMPLES} sample creati con condizione: {CONDITION_VECTOR}')
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    main()
