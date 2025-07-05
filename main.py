from pathlib import Path
from typing import List, Union

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessorFast, CLIPModel


class CoSENTLoss(torch.nn.Module):
    """
    CoSENT loss adapted for geographic similarity learning.
    """

    def __init__(self, scale: float = 20.0):
        """
        Args:
            scale: Scaling factor for cosine similarities (equivalent to 1/temperature)
        """
        super().__init__()
        self.scale = scale

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        distances: torch.Tensor,
    ):
        """
        Compute CoSENT loss for pairs of embeddings.

        Args:
            embeddings1: Batch of first embeddings [batch_size, embedding_dim]
            embeddings2: Batch of second embeddings [batch_size, embedding_dim]
            distances: Geographic distances for each pair [batch_size]

        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarities for each pair
        # similarities[i] = cos(embed1[i], embed2[i])
        similarities = (embeddings1 * embeddings2).sum(dim=1)

        # Scale similarities
        similarities = similarities * self.scale

        # Create similarity difference matrix
        # sim_diff[i,j] = similarity[i] - similarity[j]
        sim_diff = similarities[:, None] - similarities[None, :]

        # Create label matrix based on distances
        # labels[i,j] = 1 if distance[i] < distance[j], else 0
        # This means pair i should have higher similarity than pair j
        labels = distances[:, None] < distances[None, :]
        labels = labels.float()

        # Mask out irrelevant pairs (where distance[i] >= distance[j])
        # We only want to penalize cases where a far pair has higher similarity than a close pair
        sim_diff = sim_diff - (1 - labels) * 1e12

        # Flatten and add zero for numerical stability
        sim_diff = sim_diff.view(-1)
        sim_diff = torch.cat([torch.zeros(1, device=sim_diff.device), sim_diff])

        # Compute log-sum-exp
        loss = torch.logsumexp(sim_diff, dim=0)

        return loss


class GeoDataset(Dataset):
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, data_dir: Union[str, Path], split: str = "train"):
        """
        Initialize the GeoDataset.

        Args:
            data_dir: Root directory containing split subdirectories
            split: Dataset split name (e.g., 'train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Validate directory structure
        self._validate_directory()

        # Load image paths
        self.image_paths = self._load_image_paths()

        # Initialize processor
        self.processor = CLIPImageProcessorFast.from_pretrained("geolocal/StreetCLIP")

    def _validate_directory(self) -> None:
        """Validate that the dataset directory structure exists."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        if not self.split_dir.is_dir():
            raise NotADirectoryError(f"Split path is not a directory: {self.split_dir}")

    def _load_image_paths(self) -> List[Path]:
        """
        Load all image file paths from the split directory.

        Returns:
            List of Path objects for valid image files
        """
        image_paths = []

        # Use pathlib's glob for efficient file discovery
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = f"*{ext}"
            image_paths.extend(self.split_dir.glob(pattern))

        # Remove duplicates and sort for consistent ordering
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(
                f"No images found in {self.split_dir}. "
                f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        return image_paths

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for an image by index."""
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        return self.image_paths[idx]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        """
        Return a pair of images and their geographic distance.

        Returns:
            dict containing:
                - 'image1': First image tensor
                - 'image2': Second image tensor
                - 'distance': Geographic distance in kilometers
        """
        # Use idx as seed for reproducible random pairs
        random.seed(idx)

        # Select two different images
        n_images = len(self.image_paths)
        idx1 = random.randint(0, n_images - 1)
        idx2 = random.randint(0, n_images - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, n_images - 1)

        # Load images
        path1 = self.image_paths[idx1]
        path2 = self.image_paths[idx2]

        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")

        # Process images using CLIP processor
        processed1 = self.processor(images=image1, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        processed2 = self.processor(images=image2, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        # Extract GPS coordinates from filenames
        lat1, lon1 = ...
        lat2, lon2 = ...

        # Calculate distance
        distance = haversine_distance(lat1, lon1, lat2, lon2)

        return {
            "image1": processed1,
            "image2": processed2,
            "distance": torch.tensor(distance, dtype=torch.float32),
        }


class GeoDataModule(L.LightningDataModule):

    train_dataset: GeoDataset
    val_dataset: GeoDataset

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize the GeoDataModule.

        Args:
            data_dir: Root directory containing train/val/test subdirectories
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_dataset = GeoDataset(self.data_dir, split="train")
        self.val_dataset = GeoDataset(self.data_dir, split="val")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class GeoModel(L.LightningModule):
    """Lightning module for geographic image classification."""

    def __init__(self, learning_rate: float = 1e-4, temperature: float = 0.07):
        """
        Initialize the GeoModel.

        Args:
            learning_rate: Learning rate for the optimizer
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.loss_fn = CoSENTLoss()

        # Save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """
        Training step for geographic similarity learning.

        Args:
            batch: Dictionary containing 'image1', 'image2', and 'distance'
            batch_idx: Index of the current batch

        Returns:
            Loss value
        """
        # Extract images and distances from batch
        images1 = batch["image1"]
        images2 = batch["image2"]
        distances = batch["distance"]

        # Get image embeddings using CLIP's vision encoder
        # Note: get_image_features returns normalized embeddings by default
        embeddings1 = self.model.get_image_features(images1)
        embeddings2 = self.model.get_image_features(images2)

        # Compute CoSENT loss
        loss = self.loss_fn(embeddings1, embeddings2, distances)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - same as training but without gradient computation."""
        images1 = batch["image1"]
        images2 = batch["image2"]
        distances = batch["distance"]

        embeddings1 = self.model.get_image_features(images1)
        embeddings2 = self.model.get_image_features(images2)

        loss = self.loss_fn(embeddings1, embeddings2, distances)

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


L.seed_everything(42)

model = GeoModel(learning_rate=1e-4, temperature=0.07)
wandb_logger = WandbLogger(project="geosolver", log_model=False)

# Initialize callbacks
callbacks = [
    ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="geo-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    ),
    LearningRateMonitor(logging_interval="epoch"),
    EarlyStopping(monitor="val_loss", patience=5, mode="min"),
]

# Initialize trainer
trainer = L.Trainer(
    limit_train_batches=10,  # For sanity check
    limit_val_batches=5,  # For sanity check
    max_epochs=1,
    logger=wandb_logger,
    callbacks=callbacks,
    gradient_clip_val=1.0,  # Gradient clipping for stability
    accelerator="auto",
    devices=1,
    precision="16-mixed" if torch.cuda.is_available() else 32,
    detect_anomaly=False,
)

# Initialize data module
data_module = GeoDataModule(data_dir="data/geo", batch_size=8, num_workers=4)

# Train model
trainer.fit(model=model, datamodule=data_module)
