import random
from pathlib import Path
from typing import List, Union
from geopy.distance import geodesic
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


class ComparisonCoSENTLoss(torch.nn.Module):
    """
    CoSENT loss for comparison units.

    Each unit contains a close pair (should be similar) and a far pair (should be dissimilar).
    Loss encourages similarity(close_pair) > similarity(far_pair).

    Note: Distances are not needed as the ordering is implicit in the data structure.
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(self, embeddings: torch.Tensor):
        """
        Compute loss for comparison units.

        Args:
            embeddings: Embeddings [batch_size, 4, embedding_dim]
                       Order: [close_img1, close_img2, far_img1, far_img2]

        Returns:
            Scalar loss value
        """
        # Normalize all embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)

        # Extract close and far pairs
        close_emb1 = embeddings[:, 0]  # [batch_size, embedding_dim]
        close_emb2 = embeddings[:, 1]
        far_emb1 = embeddings[:, 2]
        far_emb2 = embeddings[:, 3]

        # Compute similarities
        sim_close = (close_emb1 * close_emb2).sum(dim=1)  # [batch_size]
        sim_far = (far_emb1 * far_emb2).sum(dim=1)  # [batch_size]

        # Scale similarities
        sim_close = sim_close * self.scale
        sim_far = sim_far * self.scale

        # CoSENT loss: penalize when sim_far > sim_close
        # We want sim_close > sim_far, so the loss penalizes sim_far - sim_close
        diff = sim_far - sim_close  # [batch_size]

        # Add zero for numerical stability and compute log-sum-exp
        diff_with_zero = torch.cat([torch.zeros(1, device=diff.device), diff])
        loss = torch.logsumexp(diff_with_zero, dim=0)

        return loss


class ComparisonDataset(Dataset):
    """
    Dataset that returns comparison units for training.

    Each unit contains a close pair and a far pair where currently
    similarity(close) < similarity(far), which violates our expectations.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        pool_size: int = 10000,
        update_frequency: int = 1000,
        sample_size: int = 10000,
    ):
        """
        Initialize the ComparisonDataset.

        Args:
            data_dir: Root directory containing split subdirectories
            split: Dataset split name
            pool_size: Maximum number of comparison units to cache
            update_frequency: Mine new violations every N samples
            sample_size: Number of pairs to evaluate when mining
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Mining parameters
        self.pool_size = pool_size
        self.update_frequency = update_frequency
        self.sample_size = sample_size

        # Mining state
        self.violation_pool = []
        self.access_count = 0
        self.model = None

        # Load image paths
        self._validate_directory()
        self.image_paths = self._load_image_paths()

        # Initialize processor
        self.processor = CLIPImageProcessorFast.from_pretrained("geolocal/StreetCLIP")

    def _validate_directory(self) -> None:
        """Validate dataset directory structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

    def _load_image_paths(self) -> List[Path]:
        """Load all image file paths."""
        image_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(self.split_dir.glob(f"*{ext}"))

        image_paths = sorted(set(image_paths))
        if not image_paths:
            raise ValueError(f"No images found in {self.split_dir}")

        return image_paths

    def set_model(self, model: L.LightningModule) -> None:
        """Set the model for mining violations."""
        self.model = model
        print(f"Model set for {self.split} dataset")

    @torch.no_grad()
    def mine_violations(self) -> None:
        """Mine violation pairs where close pairs have lower similarity than far pairs."""
        if self.model is None:
            return

        print(f"\nMining violations from {self.sample_size} pairs...")
        self.model.eval()
        device = next(self.model.parameters()).device

        # Store pairs with their similarities and distances
        pairs_data = []

        # Sample and evaluate random pairs
        for _ in range(self.sample_size):
            idx1 = random.randint(0, len(self.image_paths) - 1)
            idx2 = random.randint(0, len(self.image_paths) - 1)
            if idx1 == idx2:
                continue

            try:
                # Load and process images
                path1, path2 = self.image_paths[idx1], self.image_paths[idx2]
                img1 = Image.open(path1).convert("RGB")
                img2 = Image.open(path2).convert("RGB")

                proc1 = self.processor(images=img1, return_tensors="pt")[
                    "pixel_values"
                ].to(device)
                proc2 = self.processor(images=img2, return_tensors="pt")[
                    "pixel_values"
                ].to(device)

                # Get embeddings and similarity
                emb1 = self.model.model.get_image_features(proc1)
                emb2 = self.model.model.get_image_features(proc2)
                similarity = F.cosine_similarity(emb1, emb2).item()

                # Get distance
                lat1, lon1 = parse_gps_from_filename(path1.name)
                lat2, lon2 = parse_gps_from_filename(path2.name)
                distance = geodesic(lat1, lon1, lat2, lon2).kilometers

                pairs_data.append(
                    {
                        "idx1": idx1,
                        "idx2": idx2,
                        "similarity": similarity,
                        "distance": distance,
                    }
                )

            except Exception:
                continue

        # Find violations: close pairs with lower similarity than far pairs
        violations = []
        for i in range(len(pairs_data)):
            for j in range(i + 1, len(pairs_data)):
                pair_i = pairs_data[i]
                pair_j = pairs_data[j]

                # Check if this forms a violation
                if (
                    pair_i["distance"] < pair_j["distance"]
                    and pair_i["similarity"] < pair_j["similarity"]
                ):
                    # pair_i is close but less similar, pair_j is far but more similar
                    violations.append(
                        {
                            "close_idx1": pair_i["idx1"],
                            "close_idx2": pair_i["idx2"],
                            "far_idx1": pair_j["idx1"],
                            "far_idx2": pair_j["idx2"],
                            "gap": pair_j["similarity"] - pair_i["similarity"],
                        }
                    )
                elif (
                    pair_j["distance"] < pair_i["distance"]
                    and pair_j["similarity"] < pair_i["similarity"]
                ):
                    # pair_j is close but less similar, pair_i is far but more similar
                    violations.append(
                        {
                            "close_idx1": pair_j["idx1"],
                            "close_idx2": pair_j["idx2"],
                            "far_idx1": pair_i["idx1"],
                            "far_idx2": pair_i["idx2"],
                            "gap": pair_i["similarity"] - pair_j["similarity"],
                        }
                    )

        # Sort by gap (larger gaps = harder violations) and keep top pool_size
        violations.sort(key=lambda x: x["gap"], reverse=True)
        self.violation_pool = violations[: self.pool_size]

        print(f"Found {len(self.violation_pool)} violations")
        if self.violation_pool:
            print(
                f"Similarity gap range: {self.violation_pool[-1]['gap']:.3f} to {self.violation_pool[0]['gap']:.3f}"
            )

        self.model.train()

    def __len__(self) -> int:
        """Return dataset length."""
        # Arbitrary large number for continuous training
        return len(self.image_paths) * 10

    def __getitem__(self, idx: int):
        """
        Return a comparison unit.

        Returns:
            dict with:
                - images: tensor [4, 3, H, W] containing [close_img1, close_img2, far_img1, far_img2]
                - distances: tensor [2] containing [close_distance, far_distance]
        """
        self.access_count += 1

        # Mine violations periodically
        if self.model is not None and self.access_count % self.update_frequency == 0:
            self.mine_violations()

        # Select violation or random pairs if pool is empty
        if self.violation_pool:
            violation = random.choice(self.violation_pool)
            indices = [
                violation["close_idx1"],
                violation["close_idx2"],
                violation["far_idx1"],
                violation["far_idx2"],
            ]
        else:
            # Random fallback before first mining
            indices = [random.randint(0, len(self.image_paths) - 1) for _ in range(4)]

        # Load and process images
        images = []
        for idx in indices:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            processed = self.processor(images=img, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            images.append(processed)

        # Stack images
        images_tensor = torch.stack(images)  # [4, 3, H, W]
        return {"images": images_tensor}


class GeoDataset(Dataset):
    """Standard dataset for validation - returns random pairs."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, data_dir: Union[str, Path], split: str = "val"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        self._validate_directory()
        self.image_paths = self._load_image_paths()
        self.processor = CLIPImageProcessorFast.from_pretrained("geolocal/StreetCLIP")

    def _validate_directory(self) -> None:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

    def _load_image_paths(self) -> List[Path]:
        image_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(self.split_dir.glob(f"*{ext}"))
        return sorted(set(image_paths))

    def __len__(self) -> int:
        return len(self.image_paths) * 10

    def __getitem__(self, idx: int):
        # Random pair selection
        random.seed(idx)
        n_images = len(self.image_paths)
        idx1 = random.randint(0, n_images - 1)
        idx2 = random.randint(0, n_images - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, n_images - 1)

        # Load images
        img1 = Image.open(self.image_paths[idx1]).convert("RGB")
        img2 = Image.open(self.image_paths[idx2]).convert("RGB")

        # Process images
        proc1 = self.processor(images=img1, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        proc2 = self.processor(images=img2, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        # Get distance
        lat1, lon1 = parse_gps_from_filename(self.image_paths[idx1].name)
        lat2, lon2 = parse_gps_from_filename(self.image_paths[idx2].name)
        distance = geodesic(lat1, lon1, lat2, lon2).kilometers

        return {
            "image1": proc1,
            "image2": proc2,
            "distance": torch.tensor(distance, dtype=torch.float32),
        }


class GeoDataModule(L.LightningDataModule):
    """Data module with comparison-based training and random validation."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        pool_size: int = 10000,
        update_frequency: int = 1000,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pool_size = pool_size
        self.update_frequency = update_frequency

    def setup(self, stage=None):
        # Training uses comparison units
        self.train_dataset = ComparisonDataset(
            self.data_dir,
            split="train",
            pool_size=self.pool_size,
            update_frequency=self.update_frequency,
        )

        # Validation uses random pairs
        self.val_dataset = GeoDataset(self.data_dir, split="val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class GeoModel(L.LightningModule):
    """Model for geographic similarity learning."""

    def __init__(self, learning_rate: float = 1e-4, scale: float = 20.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.scale = scale
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.loss_fn = ComparisonCoSENTLoss(scale=scale)
        self.val_loss_fn = CoSENTLoss(scale=scale)  # Standard loss for validation

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Training step with comparison units."""
        images = batch["images"]  # [batch_size, 4, 3, H, W]

        # Reshape for efficient processing
        batch_size = images.shape[0]
        images_flat = images.view(
            batch_size * 4, *images.shape[2:]
        )  # [batch_size*4, 3, H, W]

        # Get all embeddings in one forward pass
        embeddings_flat = self.model.get_image_features(
            images_flat
        )  # [batch_size*4, embedding_dim]

        # Reshape back to [batch_size, 4, embedding_dim]
        embeddings = embeddings_flat.view(batch_size, 4, -1)

        # Compute loss
        loss = self.loss_fn(embeddings)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with random pairs."""
        images1 = batch["image1"]
        images2 = batch["image2"]
        distances = batch["distance"]

        embeddings1 = self.model.get_image_features(images1)
        embeddings2 = self.model.get_image_features(images2)

        loss = self.val_loss_fn(embeddings1, embeddings2, distances)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


# Original CoSENT loss for validation
class CoSENTLoss(torch.nn.Module):
    """Standard CoSENT loss for validation."""

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        distances: torch.Tensor,
    ):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        similarities = (embeddings1 * embeddings2).sum(dim=1)
        similarities = similarities * self.scale

        sim_diff = similarities[:, None] - similarities[None, :]
        labels = distances[:, None] < distances[None, :]
        labels = labels.float()

        sim_diff = sim_diff - (1 - labels) * 1e12
        sim_diff = sim_diff.view(-1)
        sim_diff = torch.cat([torch.zeros(1, device=sim_diff.device), sim_diff])

        return torch.logsumexp(sim_diff, dim=0)


if __name__ == "__main__":
    L.seed_everything(42)

    # Initialize model
    model = GeoModel(learning_rate=1e-4, scale=20.0)

    # Initialize data module
    data_module = GeoDataModule(
        data_dir="data/geo",
        batch_size=8,
        num_workers=4,
        pool_size=10000,  # Cache up to 10k violations
        update_frequency=1000,  # Re-mine every 1000 samples
    )

    # Setup datasets
    data_module.setup()

    # Link model to training dataset
    data_module.train_dataset.set_model(model)

    # Initialize logger
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
        limit_train_batches=10,
        limit_val_batches=5,
        max_epochs=1,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        detect_anomaly=False,
    )

    # Train model
    trainer.fit(model=model, datamodule=data_module)
