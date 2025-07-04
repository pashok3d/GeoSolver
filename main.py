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

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Get a preprocessed image by index.

        Args:
            idx: Index of the image to retrieve

        Returns:
            Preprocessed image tensor with shape (C, H, W)
        """
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        image_path = self.image_paths[idx]

        try:
            # Load and convert image
            image = Image.open(image_path).convert("RGB")

            # Process image
            inputs = self.processor(images=image, return_tensors="pt")

            # Return tensor without batch dimension
            return inputs.pixel_values.squeeze(0)

        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for an image by index."""
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        return self.image_paths[idx]


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

        # Save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pixel_values = batch

        # Get image embeddings
        image_features = self.model.get_image_features(pixel_values)

        # Normalize features for cosine similarity
        image_features = F.normalize(image_features, p=2, dim=1)

        # Simple contrastive loss within batch (SimCLR-style)
        # This treats each image as its own class
        batch_size = image_features.shape[0]

        # Compute similarity matrix
        similarity_matrix = (
            torch.matmul(image_features, image_features.T) / self.temperature
        )

        # Create labels (each image is its own class)
        labels = torch.arange(batch_size, device=self.device)

        loss = F.cross_entropy(similarity_matrix, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch

        # Get image embeddings
        image_features = self.model.get_image_features(pixel_values)

        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)

        # Same loss as training
        batch_size = image_features.shape[0]
        similarity_matrix = (
            torch.matmul(image_features, image_features.T) / self.temperature
        )
        labels = torch.arange(batch_size, device=self.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

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
    precision="16-mixed",
    detect_anomaly=False,
)

# Initialize data module
data_module = GeoDataModule(data_dir="data/geo", batch_size=8, num_workers=4)

# Train model
trainer.fit(model=model, datamodule=data_module)
