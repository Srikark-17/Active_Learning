from fastapi import FastAPI, UploadFile, HTTPException, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List, Tuple
import json
import time
import types
import asyncio
import glob
import shutil
import tempfile
import traceback
from model_utils import safe_load_model

class ProjectInit(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    project_name: str
    model_type: Literal["resnet18", "resnet50"] = Field(
        description="Type of model to use (resnet18 or resnet50)"
    )
    num_classes: int = Field(gt=0, description="Number of classes to classify")
    val_split: float = Field(
        default=0.2, 
        gt=0.0, 
        lt=1.0, 
        description="Validation split ratio (0-1)"
    )
    initial_labeled_ratio: float = Field(
        default=0.1, 
        gt=0.0, 
        lt=1.0, 
        description="Initial labeled data ratio (0-1)"
    )
    # Add training configuration fields
    sampling_strategy: str = Field(
        default="least_confidence",
        description="Active learning sampling strategy"
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Training batch size"
    )
    epochs: int = Field(
        default=10,
        gt=0,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Initial learning rate"
    )

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    sampling_strategy: str
    learning_rate: float = 0.001

class BatchRequest(BaseModel):
    strategy: str
    batch_size: int

class LabelSubmission(BaseModel):
    image_id: int
    label: int

class ActiveLearningManager:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labeled_data = {}  # {image_id: (image_tensor, label)}
        self.unlabeled_data = {}  # {image_id: image_tensor}
        self.validation_data = {}  # {image_id: (image_tensor, label)}
        self.current_batch = []
        self.episode = 0
        self.project_name = None
        self.output_dir = None
        self.checkpoint_manager = None
        self.lr_scheduler = None
        self.lr_config = {
            'strategy': 'plateau',
            'initial_lr': 0.001,
            'factor': 0.1,
            'patience': 5,
            'min_lr': 1e-6
        }
        
        # Tracking metrics
        self.plot_episode_xvalues = []
        self.plot_episode_yvalues = []
        self.plot_epoch_xvalues = []
        self.plot_epoch_yvalues = []
        self.best_val_acc = 0
        self.best_model_state = None
        self.training_config = {
        'sampling_strategy': 'least_confidence',  # default strategy
        'batch_size': 32,  # default batch size
        'epochs': 10,  # default epochs
        'learning_rate': 0.001,  # default learning rate
        'scheduler': {
            'strategy': 'plateau',  # default scheduler strategy
            'params': {
                'mode': 'max',
                'factor': 0.1,
                'patience': 5,
                'verbose': True,
                'min_lr': 1e-6
            }
        }
    }

        self.config = {
            'val_split': 0.2,  # 20% validation like original script
            'initial_labeled_ratio': 0.1,  # 10% initial labeled data
        }
        
        # Training history
        self.episode_history = []  # List of dicts with episode metrics
        
        # Default transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def initialize_project(self, project_name: str, model_name: str, num_classes: int, config: dict = None):
        try:
            # Validation checks with more flexibility
            if not project_name:
                raise ValueError("Project name is required")
                
            # More flexible model type handling
            valid_models = ["resnet18", "resnet50", "efficientnet", "densenet", "mobilenet", "custom"]
            if model_name not in valid_models:
                print(f"Warning: Unknown model type '{model_name}'. Treating as 'custom'")
                model_name = "custom"  # Fallback to custom
                    
            # Set up project
            self.project_name = project_name
            self.output_dir = os.path.join("output", project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.output_dir, exist_ok=True)
                
            # Initialize model based on type
            if model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
            elif model_name == "efficientnet":
                # Use ResNet50 as a fallback but print a notice
                print("EfficientNet not directly supported - using ResNet50 as base architecture")
                self.model = models.resnet50(pretrained=True)
            elif model_name == "densenet":
                # Use ResNet50 as a fallback but print a notice
                print("DenseNet not directly supported - using ResNet50 as base architecture")
                self.model = models.resnet50(pretrained=True)
            elif model_name == "mobilenet":
                # Use ResNet50 as a fallback but print a notice
                print("MobileNet not directly supported - using ResNet50 as base architecture")
                self.model = models.resnet50(pretrained=True)
            elif model_name == "custom":
                # For custom models, use ResNet50 as the base architecture
                print("Using ResNet50 as base architecture for custom model")
                self.model = models.resnet50(pretrained=True)
            else:
                # Should never reach here due to validation above, but just in case
                print(f"Unsupported model type: {model_name}, using ResNet50")
                self.model = models.resnet50(pretrained=True)
                
            # Modify final layer for our number of classes
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            self.model = self.model.to(self.device)

            # Update training config if provided
            if config:
                self.training_config.update(config)

            # Initialize optimizer after model
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config['learning_rate']
            )
            
            # Initialize LR scheduler with proper configuration
            scheduler_config = self.training_config.get('scheduler', {
                'strategy': 'plateau',
                'params': {
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': 5,
                    'verbose': True,
                    'min_lr': 1e-6
                }
            })
            
            self.lr_scheduler = LRSchedulerManager(
                optimizer=self.optimizer,
                strategy=scheduler_config['strategy'],
                initial_lr=self.training_config['learning_rate'],
                **scheduler_config['params']
            )
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(self.output_dir)
                
            return {
                "status": "success",
                "output_dir": self.output_dir,
                "config": self.training_config
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    def save_state(self, is_best: bool = False):
        """Save complete model and training state"""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized")
            
        state = {
            'episode': self.episode,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_config': self.training_config,
            'labeled_indices': list(self.labeled_data.keys()),
            'unlabeled_indices': list(self.unlabeled_data.keys()),
            'validation_indices': list(self.validation_data.keys()),
            'metrics': {
                'episode_accuracies': {
                    'x': self.plot_episode_xvalues,
                    'y': self.plot_episode_yvalues
                },
                'epoch_losses': {
                    'x': self.plot_epoch_xvalues,
                    'y': self.plot_epoch_yvalues
                }
            },
            'episode_history': self.episode_history
        }
        
        return self.checkpoint_manager.save_checkpoint(state, is_best)

    async def add_initial_data(self, files: List[UploadFile], val_split: float = None):
        """Add initial dataset and split into labeled/unlabeled/validation"""
        if val_split is not None:
            self.config['val_split'] = val_split

        # First, load all images
        all_data = {}
        for img_file in files:
            content = await img_file.read()
            img = Image.open(io.BytesIO(content)).convert('RGB')
            img_tensor = self.transform(img)
            img_id = len(all_data)
            all_data[img_id] = img_tensor

        # Calculate split sizes
        total_images = len(all_data)
        val_size = int(total_images * self.config['val_split'])
        initial_labeled_size = int((total_images - val_size) * self.config['initial_labeled_ratio'])

        # Create random splits
        all_indices = list(all_data.keys())
        np.random.shuffle(all_indices)

        # Split indices
        val_indices = all_indices[:val_size]
        initial_labeled_indices = all_indices[val_size:val_size + initial_labeled_size]
        unlabeled_indices = all_indices[val_size + initial_labeled_size:]

        # FIXED: Properly assign data to sets
        
        # 1. Put unlabeled data in unlabeled set
        for idx in unlabeled_indices:
            self.unlabeled_data[idx] = all_data[idx]

        # 2. Put initial labeled data in unlabeled set (they'll be moved to labeled when user labels them)
        for idx in initial_labeled_indices:
            self.unlabeled_data[idx] = all_data[idx]

        # 3. Put validation data in validation set WITH TEMPORARY LABELS (for validation)
        # In active learning, we typically assign random labels to validation set initially
        # or use a small subset of pre-labeled data
        for idx in val_indices:
            # For now, assign random labels to validation set so we can compute validation accuracy
            # In a real scenario, you'd want some pre-labeled validation data
            temp_label = np.random.randint(0, self.model.fc.out_features if hasattr(self.model, 'fc') else 2)
            self.validation_data[idx] = (all_data[idx], temp_label)

        # Save split information
        split_info = {
            "total_images": total_images,
            "validation": len(self.validation_data),
            "initial_labeled": 0,  # Initially 0, will grow as user labels
            "unlabeled": len(self.unlabeled_data)
        }

        return split_info

    def get_next_batch(self, strategy: str, batch_size: int) -> List[dict]:
        """
        Select next batch of samples using specified strategy with improved error handling
        Args:
            strategy: Sampling strategy to use
            batch_size: Number of samples to select
        Returns:
            List of selected samples with metadata
        """
        if not self.model:
            raise HTTPException(status_code=400, detail="Model not initialized")

        if len(self.unlabeled_data) == 0:
            raise HTTPException(status_code=400, detail="No unlabeled data available")
            
        # Ensure batch size isn't too large
        if batch_size > len(self.unlabeled_data):
            print(f"Warning: Requested batch size {batch_size} is larger than available unlabeled data ({len(self.unlabeled_data)})")
            batch_size = len(self.unlabeled_data)
            
        # Add safety check for very small batch sizes
        if batch_size <= 0:
            batch_size = min(32, len(self.unlabeled_data))  # Default to 32 or smaller if not enough data
            
        try:
            # Get uncertainty scores for all unlabeled samples
            sample_scores = self._get_sample_scores(strategy)
            
            # Select samples based on strategy
            selected_samples = self._select_samples(sample_scores, batch_size, strategy)
            
            # Update current batch
            self.current_batch = [x["image_id"] for x in selected_samples]
            
            return selected_samples
        except Exception as e:
            print(f"Error in get_next_batch: {str(e)}")
            traceback.print_exc()
            
            # Try with random sampling as fallback
            if strategy != "random":
                print("Falling back to random sampling strategy")
                try:
                    # Implement simple random sampling directly
                    image_ids = list(self.unlabeled_data.keys())
                    selected_ids = random.sample(image_ids, min(batch_size, len(image_ids)))
                    
                    selected_samples = []
                    for img_id in selected_ids:
                        # Get image tensor
                        img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                        
                        # Get model predictions without trying to compute uncertainty
                        with torch.no_grad():
                            try:
                                outputs = self.model(img_tensor)
                                probs = torch.softmax(outputs, dim=1)
                                
                                # Create prediction list
                                predictions = [
                                    {"label": f"Label {i}", "confidence": float(p)} 
                                    for i, p in enumerate(probs[0])
                                ]
                                
                                selected_samples.append({
                                    "image_id": img_id,
                                    "uncertainty": 0.5,  # Default uncertainty
                                    "predictions": predictions
                                })
                            except Exception as inner_e:
                                # If even simple prediction fails, use minimal info
                                print(f"Error making predictions: {str(inner_e)}")
                                selected_samples.append({
                                    "image_id": img_id,
                                    "uncertainty": 0.5,  # Default uncertainty
                                    "predictions": [{"label": "Unknown", "confidence": 0.0}]
                                })
                    
                    self.current_batch = [x["image_id"] for x in selected_samples]
                    return selected_samples
                except Exception as fallback_e:
                    print(f"Fallback strategy also failed: {str(fallback_e)}")
                    traceback.print_exc()
                    raise
            else:
                # If we're already using random sampling and it failed, re-raise
                raise
    
    def _get_sample_scores(self, strategy: str) -> List[Tuple[int, float, List[dict]]]:
        """
        Compute scores for all unlabeled samples
        Returns:
            List of tuples (image_id, uncertainty_score, predictions)
        """
        sample_scores = []
        self.model.eval()
        
        with torch.no_grad():
            for img_id, img_tensor in self.unlabeled_data.items():
                # Get model outputs
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get features if needed
                features = self._get_features(img_tensor) if strategy == "diversity" else None
                
                # Compute uncertainty
                uncertainty = self._compute_uncertainty(probs, strategy, features)
                
                # Get predictions
                predictions = [
                    {"label": f"Label {i}", "confidence": float(p)} 
                    for i, p in enumerate(probs[0])
                ]
                
                sample_scores.append((img_id, uncertainty, predictions))
                
        return sample_scores

    def _select_samples(self, sample_scores: List[Tuple[int, float, List[dict]]], 
                   batch_size: int, strategy: str) -> List[dict]:
        """
        Select batch_size samples based on scores and strategy
        """
        if strategy == "diversity":
            # For diversity, we want to maximize coverage
            selected = self._select_diverse_samples(sample_scores, batch_size)
        else:
            # For uncertainty-based strategies, sort by score
            sample_scores.sort(key=lambda x: x[1], reverse=True)
            selected = sample_scores[:batch_size]
        
        # Format selected samples
        return [
            {
                "image_id": img_id,
                "uncertainty": score,
                "predictions": preds
            }
            for img_id, score, preds in selected
        ]

    def _select_diverse_samples(self, sample_scores: List[Tuple[int, float, List[dict]]], 
                          batch_size: int) -> List[Tuple[int, float, List[dict]]]:
        """
        Select diverse samples using greedy approach
        """
        if batch_size >= len(sample_scores):
            return sample_scores
            
        selected = []
        remaining = sample_scores.copy()
        
        # Select highest uncertainty sample first
        remaining.sort(key=lambda x: x[1], reverse=True)
        selected.append(remaining.pop(0))
        
        # Greedily select rest based on maximum distance to selected set
        while len(selected) < batch_size and remaining:
            # Get features for all remaining samples
            remaining_features = []
            for _, _, preds in remaining:
                probs = torch.tensor([[p["confidence"] for p in preds]])
                remaining_features.append(probs)
            remaining_features = torch.cat(remaining_features, dim=0)
            
            # Get features for selected samples
            selected_features = []
            for _, _, preds in selected:
                probs = torch.tensor([[p["confidence"] for p in preds]])
                selected_features.append(probs)
            selected_features = torch.cat(selected_features, dim=0)
            
            # Compute distances
            distances = torch.cdist(remaining_features, selected_features)
            min_distances = distances.min(dim=1)[0]
            
            # Select sample with maximum minimum distance
            best_idx = min_distances.argmax().item()
            selected.append(remaining.pop(best_idx))
        
        return selected

    def train_epoch(self, optimizer, criterion, batch_size=32):
        """Train for one epoch with proper validation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Prepare training data
        all_images = []
        all_labels = []
        for img_tensor, label in self.labeled_data.values():
            all_images.append(img_tensor)
            all_labels.append(label)

        if len(all_images) == 0:
            raise ValueError("No labeled data available for training")

        X = torch.stack(all_images)
        y = torch.tensor(all_labels)

        # Training loop
        indices = torch.randperm(len(all_images))
        batch_losses = []

        for i in range(0, len(all_images), batch_size):
            batch_indices = indices[i:min(i + batch_size, len(all_images))]
            batch_X = X[batch_indices].to(self.device)
            batch_y = y[batch_indices].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy
    
    def validate_model(self):
        """Perform validation on the validation set"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        # Only use labeled validation samples
        labeled_validation = {
            idx: (tensor, label) 
            for idx, (tensor, label) in self.validation_data.items() 
            if label is not None
        }
        
        if len(labeled_validation) == 0:
            return 0.0
            
        batch_size = 32
        with torch.no_grad():
            for idx in labeled_validation:
                img_tensor, label = labeled_validation[idx]
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += 1
                
        validation_accuracy = 100.0 * total_correct / total_samples
        return validation_accuracy

    def validate(self):
        """Validate model performance - improved for active learning"""
        try:
            if len(self.validation_data) == 0:
                print("Warning: No validation data available")
                return 0.0
                    
            # Get labeled validation data
            labeled_validation = [
                (img, label) for img, label in self.validation_data.values() 
                if label is not None
            ]
            
            # If no labeled validation data, we can't compute accuracy
            if len(labeled_validation) == 0:
                print(f"Warning: No labeled validation data (0/{len(self.validation_data)} samples labeled)")
                print("Validation data exists but needs labels. Consider labeling some validation samples.")
                return 0.0

            self.model.eval()
            total_correct = 0
            total_samples = 0

            batch_size = 32
            with torch.no_grad():
                for i in range(0, len(labeled_validation), batch_size):
                    batch = labeled_validation[i:i + batch_size]
                    images = torch.stack([img for img, _ in batch]).to(self.device)
                    labels = torch.tensor([label for _, label in batch]).to(self.device)

                    outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            val_accuracy = 100.0 * total_correct / total_samples
            print(f"Validation Accuracy: {val_accuracy:.2f}% ({total_correct}/{total_samples})")
            return val_accuracy

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return 0.0

    def train(self, epochs: int, batch_size: int, learning_rate: float):
        """Train model on labeled data"""
        try:
            if len(self.labeled_data) == 0:
                raise HTTPException(status_code=400, detail="No labeled data available")

            if len(self.validation_data) == 0:
                raise HTTPException(status_code=400, detail="No validation data available")

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0
            best_model = None
            
            for epoch in range(epochs):
                try:
                    train_loss, train_acc = self.train_epoch(optimizer, criterion)
                    val_acc = self.validate()

                    # Save metrics for plotting
                    self.plot_epoch_xvalues.append(epoch + 1)
                    self.plot_epoch_yvalues.append(train_loss)

                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = self.model.state_dict().copy()

                    # Plot training progress
                    self.plot_training_progress(epoch + 1, train_loss, val_acc)
                except Exception as e:
                    print(f"Error in epoch {epoch}: {str(e)}")
                    raise

            # Save episode results
            self.plot_episode_xvalues.append(self.episode)
            self.plot_episode_yvalues.append(best_val_acc)
            
            # Increment episode
            self.episode += 1

            return {
                "status": "success",
                "epochs_completed": epochs,
                "final_accuracy": val_acc,
                "best_accuracy": best_val_acc
            }
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def plot_training_progress(self, epoch, loss, accuracy):
        """Plot and save training progress"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.plot_epoch_xvalues, self.plot_epoch_yvalues)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss (Episode {self.episode})")
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.plot_episode_xvalues, self.plot_episode_yvalues)
        plt.xlabel("Episode")
        plt.ylabel("Validation Accuracy")
        plt.title("Active Learning Progress")
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"progress_ep{self.episode}_e{epoch}.png"))
        plt.close()

    def submit_label(self, image_id: int, label: int):
        """Submit label for an image"""
        # Check where the image belongs
        if image_id in self.validation_data:
            img_tensor = self.validation_data[image_id][0]
            self.validation_data[image_id] = (img_tensor, label)
        elif image_id in self.unlabeled_data:
            img_tensor = self.unlabeled_data.pop(image_id)
            self.labeled_data[image_id] = (img_tensor, label)
        else:
            raise HTTPException(status_code=400, detail="Image not found")
        
        return {
            "status": "success",
            "labeled_count": len(self.labeled_data),
            "unlabeled_count": len(self.unlabeled_data),
            "validation_count": len([x for x in self.validation_data.values() if x[1] is not None])
        }
    
    def train_episode(self, epochs: int, batch_size: int, learning_rate: float):
        """Run a complete training episode with improved batch selection, checkpointing, and LR scheduling"""
        try:
            if len(self.labeled_data) == 0:
                raise ValueError("No labeled data available for training")

            # Initialize checkpoint manager if not exists
            if self.checkpoint_manager is None:
                self.checkpoint_manager = CheckpointManager(self.output_dir)


            # Initialize optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Initialize scheduler with configurable strategy
            scheduler_config = self.training_config.get('scheduler', {
                'strategy': 'plateau',  # default strategy
                'params': {
                    'mode': 'max',
                    'factor': 0.1,
                    'patience': 5,
                    'verbose': True,
                    'min_lr': 1e-6
                }
            })
            
            # Create scheduler based on strategy
            if scheduler_config['strategy'] == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_config['params']
                )
            elif scheduler_config['strategy'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                    eta_min=scheduler_config['params'].get('min_lr', 0)
                )
            elif scheduler_config['strategy'] == 'warmup':
                steps_per_epoch = len(self.labeled_data) // batch_size
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_config['params'].get('max_lr', learning_rate * 10),
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=scheduler_config['params'].get('warmup_pct', 0.3)
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.1, patience=5, verbose=True
                )

            criterion = nn.CrossEntropyLoss()
            best_val_acc = 0
            best_model_state = None
            lr_history = []

            # Training loop with validation
            for epoch in range(epochs):
                # Train for one epoch
                train_loss, train_acc = self.train_epoch(optimizer, criterion, batch_size)
                
                # Validate
                val_acc = self.validate_model()
                
                # Update learning rate based on scheduler strategy
                current_lr = optimizer.param_groups[0]['lr']
                if scheduler_config['strategy'] == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
                
                # Record LR change
                new_lr = optimizer.param_groups[0]['lr']
                lr_history.append({
                    'epoch': epoch + 1,
                    'old_lr': current_lr,
                    'new_lr': new_lr,
                    'val_acc': val_acc
                })
                
                # Save checkpoint if best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    
                    try:
                        if self.checkpoint_manager:
                            state = {
                                'episode': self.episode,
                                'model_state': best_model_state,
                                'optimizer_state': optimizer.state_dict(),
                                'scheduler_state': scheduler.state_dict(),
                                'scheduler_config': scheduler_config,
                                'best_val_acc': best_val_acc,
                                'training_config': self.training_config,
                                'labeled_indices': list(self.labeled_data.keys()),
                                'unlabeled_indices': list(self.unlabeled_data.keys()),
                                'validation_indices': list(self.validation_data.keys()),
                                'lr_history': lr_history,
                                'metrics': {
                                    'episode_accuracies': {
                                        'x': self.plot_episode_xvalues,
                                        'y': self.plot_episode_yvalues
                                    },
                                    'epoch_losses': {
                                        'x': self.plot_epoch_xvalues,
                                        'y': self.plot_epoch_yvalues
                                    }
                                },
                                'episode_history': self.episode_history
                            }
                            self.checkpoint_manager.save_checkpoint(state, is_best=True)
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {str(e)}")
                        # Continue training even if checkpoint saving fails
                        pass


                # Store training progress
                self.plot_epoch_xvalues.append(epoch + 1)
                self.plot_epoch_yvalues.append(train_loss)

                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Training Accuracy: {train_acc:.2f}%")
                print(f"Validation Accuracy: {val_acc:.2f}%")
                print(f"Learning Rate: {new_lr:.6f}")

            # Restore best model state
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                self.best_val_acc = best_val_acc
                self.best_model_state = best_model_state

            train_result = {
                "status": "success",
                "final_accuracy": val_acc,
                "best_accuracy": best_val_acc,
                "lr_history": lr_history
            }

            # Select next batch if training successful
            try:
                # Get next batch using current model
                next_batch = self.get_next_batch(
                    strategy=self.training_config["sampling_strategy"],
                    batch_size=batch_size
                )
                
                # Update episode metrics
                episode_metrics = {
                    'episode': self.episode,
                    'train_result': train_result,
                    'batch_size': len(next_batch),
                    'strategy': self.training_config["sampling_strategy"],
                    'labeled_size': len(self.labeled_data),
                    'unlabeled_size': len(self.unlabeled_data),
                    'best_val_acc': best_val_acc,
                    'learning_rate': new_lr,
                    'lr_history': lr_history
                }
                
                self.episode_history.append(episode_metrics)
                
                # Save episode checkpoint
                if hasattr(self, 'checkpoint_manager'):
                    state = {
                        'episode': self.episode,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'scheduler_config': scheduler_config,
                        'best_val_acc': best_val_acc,
                        'training_config': self.training_config,
                        'labeled_indices': list(self.labeled_data.keys()),
                        'unlabeled_indices': list(self.unlabeled_data.keys()),
                        'validation_indices': list(self.validation_data.keys()),
                        'lr_history': lr_history,
                        'metrics': {
                            'episode_accuracies': {
                                'x': self.plot_episode_xvalues,
                                'y': self.plot_episode_yvalues
                            },
                            'epoch_losses': {
                                'x': self.plot_epoch_xvalues,
                                'y': self.plot_epoch_yvalues
                            }
                        },
                        'episode_history': self.episode_history
                    }
                    self.checkpoint_manager.save_checkpoint(state)

                self.episode += 1
                
                return {
                    "status": "success",
                    "metrics": episode_metrics,
                    "next_batch": next_batch
                }
                    
            except Exception as e:
                raise ValueError(f"Error selecting next batch: {str(e)}")
                
        except Exception as e:
            print(f"Error in train_episode: {str(e)}")
            raise

    def load_state(self, checkpoint_path: str = None):
        """Load complete model and training state"""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized")
            
        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.model, 
            self.optimizer, 
            self.scheduler,
            checkpoint_path
        )
        
        if checkpoint:
            # Restore training state
            self.episode = checkpoint['episode']
            self.best_val_acc = checkpoint['best_val_acc']
            self.training_config = checkpoint['training_config']
            
            # Restore data indices
            self.restore_data_splits(
                checkpoint['labeled_indices'],
                checkpoint['unlabeled_indices'],
                checkpoint['validation_indices']
            )
            
            # Restore metrics
            metrics = checkpoint['metrics']
            self.plot_episode_xvalues = metrics['episode_accuracies']['x']
            self.plot_episode_yvalues = metrics['episode_accuracies']['y']
            self.plot_epoch_xvalues = metrics['epoch_losses']['x']
            self.plot_epoch_yvalues = metrics['epoch_losses']['y']
            
            self.lr_config = checkpoint['lr_config']
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

            # Restore episode history
            self.episode_history = checkpoint['episode_history']
            
        return checkpoint

    def get_active_learning_batch(self, strategy: str, batch_size: int):
        """Enhanced active learning sampling with multiple strategies"""
        if strategy not in ["entropy", "margin", "least_confidence", "diversity", "random"]:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        if strategy == "random":
            return self._random_sampling(batch_size)
            
        uncertainties = []
        self.model.eval()
        
        with torch.no_grad():
            for img_id, img_tensor in self.unlabeled_data.items():
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                uncertainty = self._compute_uncertainty(probs, strategy, img_tensor)
                uncertainties.append((img_id, uncertainty))
                
        # Sort by uncertainty and select batch
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_batch = uncertainties[:batch_size]
        
        return self._prepare_batch_info(selected_batch)

    def _compute_uncertainty(self, probs: torch.Tensor, strategy: str, features: torch.Tensor = None) -> float:
        """
        Compute uncertainty score based on chosen strategy
        Args:
            probs: softmax probabilities from model
            strategy: sampling strategy to use
            features: feature representations (only needed for diversity strategy)
        Returns:
            uncertainty score between 0 and 1
        """
        try:
            if strategy == "least_confidence":
                # 1 - max probability
                return float(1 - torch.max(probs).item())
                
            elif strategy == "margin":
                # Difference between top two probabilities
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                return float(sorted_probs[0][0] - sorted_probs[0][1]).item()
                
            elif strategy == "entropy":
                # Entropy of probability distribution
                eps = 1e-10
                probs = probs.clamp(min=eps, max=1-eps)
                entropy = -(probs * torch.log(probs)).sum(dim=1)
                return float(entropy.item())
                
            elif strategy == "diversity":
                if features is None:
                    raise ValueError("Features required for diversity sampling")
                # Compute diversity using feature space distance from labeled pool
                return self._compute_diversity_score(features)
                
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")
                
        except Exception as e:
            print(f"Error computing uncertainty: {str(e)}")
            return 0.0

    def _compute_diversity_score(self, features: torch.Tensor) -> float:
        """
        Compute diversity score based on feature space distance from labeled pool
        """
        if len(self.labeled_data) == 0:
            return 1.0
            
        try:
            # Get features of labeled samples
            labeled_features = []
            self.model.eval()
            with torch.no_grad():
                for img_tensor, _ in self.labeled_data.values():
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    # Get features before final layer
                    feat = self._get_features(img_tensor)
                    labeled_features.append(feat)
                    
            labeled_features = torch.cat(labeled_features, dim=0)
            
            # Compute average distance to labeled pool
            distances = torch.cdist(features, labeled_features)
            diversity_score = distances.mean().item()
            
            # Normalize to [0,1]
            return min(1.0, diversity_score / 10.0)  # Scale factor of 10 is arbitrary
            
        except Exception as e:
            print(f"Error computing diversity score: {str(e)}")
            return 1.0

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model's penultimate layer"""
        if not hasattr(self.model, 'get_features'):
            # If model doesn't have feature extraction, add it
            original_forward = self.model.forward
            
            def get_features(self, x: torch.Tensor) -> torch.Tensor:
                # Remove the last layer temporarily
                original_fc = self.fc
                self.fc = nn.Identity()
                features = self(x)
                self.fc = original_fc
                return features
                
            self.model.get_features = types.MethodType(get_features, self.model)
        
        return self.model.get_features(x)

    def _random_sampling(self, batch_size: int):
        """Implement random sampling strategy"""
        available_ids = list(self.unlabeled_data.keys())
        selected_ids = np.random.choice(
            available_ids, 
            size=min(batch_size, len(available_ids)), 
            replace=False
        )
        return [(id, 0.0) for id in selected_ids]  # 0.0 as placeholder uncertainty

    def _prepare_batch_info(self, selected_batch):
        """Prepare detailed batch information including predictions"""
        batch_info = []
        for img_id, uncertainty in selected_batch:
            # Get model predictions
            img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
            predictions = [
                {"label": f"Label {i}", "confidence": float(p)} 
                for i, p in enumerate(probs[0])
            ]
            
            batch_info.append({
                "image_id": img_id,
                "uncertainty": uncertainty,
                "predictions": predictions
            })
            
        return batch_info

    def save_checkpoint(self):
        """Save episode checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'model_state': self.best_model_state,
            'best_val_acc': self.best_val_acc,
            'metrics': {
                'episode_accuracies': {
                    'x': self.plot_episode_xvalues,
                    'y': self.plot_episode_yvalues
                },
                'episode_history': self.episode_history
            }
        }
        
        checkpoint_path = os.path.join(
            self.output_dir, 
            f'checkpoint_ep{self.episode}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

    def get_validation_status(self):
        """Get validation set labeling status"""
        total = len(self.validation_data)
        labeled = len([1 for _, label in self.validation_data.values() if label is not None])
        return {
            "total": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "percent_labeled": (labeled / total * 100) if total > 0 else 0
        }
    
    def adapt_transformer_to_resnet(vit_state_dict, resnet_model, num_classes=None):
        """
        Adapt a Vision Transformer (ViT) model to a ResNet architecture
        by transferring compatible weights and knowledge.
        
        Args:
            vit_state_dict: State dict from a Vision Transformer model
            resnet_model: The target ResNet model
            num_classes: Number of output classes (if known)
        
        Returns:
            Tuple of (success flag, message)
        """
        # First ensure we have the correct output size if num_classes is provided
        if num_classes is not None:
            # Resize the final layer
            in_features = resnet_model.fc.in_features
            resnet_model.fc = torch.nn.Linear(in_features, num_classes)
        
        # Initialize a dictionary to track transferred knowledge
        transferred_knowledge = {
            "visual_features": False,
            "classification_head": False
        }
        
        # Check if we can use patch embeddings as initial layers
        if 'patch_embed.proj.weight' in vit_state_dict:
            # Extract the patch embedding weights
            patch_weights = vit_state_dict['patch_embed.proj.weight']
            
            # Check if the shape is compatible with the first convolutional layer
            try:
                first_conv = None
                # Find the first conv layer in ResNet
                if hasattr(resnet_model, 'conv1'):
                    first_conv = resnet_model.conv1
                
                if first_conv is not None:
                    # Check if shapes are compatible or can be adapted
                    if patch_weights.shape[0] == first_conv.weight.shape[0]:  # Same number of output channels
                        # We can initialize with the patch embedding knowledge
                        new_weights = torch.nn.functional.interpolate(
                            patch_weights, 
                            size=first_conv.weight.shape[2:],  # Target kernel size
                            mode='bilinear'
                        )
                        first_conv.weight.data = new_weights
                        transferred_knowledge["visual_features"] = True
            except Exception as e:
                print(f"Could not transfer patch embedding knowledge: {e}")
        
        # Try to transfer classification head knowledge
        if 'head.weight' in vit_state_dict and hasattr(resnet_model, 'fc'):
            try:
                vit_head_weight = vit_state_dict['head.weight']
                vit_head_bias = vit_state_dict.get('head.bias', None)
                
                # Check if output dimensions match
                if vit_head_weight.shape[0] == resnet_model.fc.weight.shape[0]:
                    # If input dimensions don't match, we can use a simple projection
                    if vit_head_weight.shape[1] != resnet_model.fc.weight.shape[1]:
                        # Create a simple projection matrix
                        projection = torch.zeros(
                            resnet_model.fc.weight.shape[1],
                            vit_head_weight.shape[1]
                        )
                        
                        # Identity mapping for the smaller dimension
                        min_dim = min(projection.shape[0], projection.shape[1])
                        projection[:min_dim, :min_dim] = torch.eye(min_dim)
                        
                        # Project the weights
                        new_weights = torch.matmul(vit_head_weight, projection.t())
                        resnet_model.fc.weight.data = new_weights
                    else:
                        # Direct transfer
                        resnet_model.fc.weight.data = vit_head_weight
                    
                    # Transfer bias if available
                    if vit_head_bias is not None and hasattr(resnet_model.fc, 'bias'):
                        resnet_model.fc.bias.data = vit_head_bias
                    
                    transferred_knowledge["classification_head"] = True
            except Exception as e:
                print(f"Could not transfer classification head knowledge: {e}")
        
        # Return success and information
        if any(transferred_knowledge.values()):
            return True, f"Adapted transformer model to ResNet. Transferred: {', '.join([k for k, v in transferred_knowledge.items() if v])}"
        else:
            return False, "Could not transfer knowledge from transformer to ResNet. Will only use initialization."

    # Then update the existing adapt_pretrained_model method:
    def adapt_pretrained_model(self, model_state, freeze_layers=True, adaptation_layers=None):
        """
        Adapt a pre-trained model for active learning by optionally freezing layers
        and preparing the model for fine-tuning.
        
        Args:
            model_state: State dict of the pretrained model
            freeze_layers: Whether to freeze early layers
            adaptation_layers: List of layer names to specifically adapt
        """
        try:
            # Check if this is a transformer-based model
            is_transformer = any(k in ['cls_token', 'pos_embed', 'patch_embed'] 
                                for k in model_state.keys())
            
            if is_transformer:
                print("Detected transformer-based model. Attempting specialized adaptation...")
                success, message = self.adapt_transformer_to_resnet(
                    model_state, 
                    self.model,
                    num_classes=self.model.fc.out_features if hasattr(self.model, 'fc') else None
                )
                if success:
                    print(f"Transformer adaptation: {message}")
                else:
                    print("Transformer adaptation failed. Falling back to standard approach.")
            
            # Continue with standard adaptation
            # Load the pretrained weights if not a transformer or transformer adaptation failed
            if not is_transformer or not success:
                if hasattr(self.model, 'load_state_dict'):
                    # Try to load directly, with a non-strict option to allow for differences
                    try:
                        self.model.load_state_dict(model_state, strict=False)
                        print("Loaded pretrained model weights (non-strict)")
                    except Exception as e:
                        print(f"Error loading model directly: {str(e)}")
                        
                        # Try to fix common key mismatches
                        fixed_state_dict = {}
                        for k, v in model_state.items():
                            # Handle module prefix differences (common with DataParallel)
                            if k.startswith('module.') and not any(key.startswith('module.') for key in self.model.state_dict()):
                                fixed_state_dict[k[7:]] = v
                            elif not k.startswith('module.') and any(key.startswith('module.') for key in self.model.state_dict()):
                                fixed_state_dict['module.' + k] = v
                            else:
                                fixed_state_dict[k] = v
                        
                        # Try loading with fixed keys
                        missing_keys, unexpected_keys = self.model.load_state_dict(fixed_state_dict, strict=False)
                        print(f"Loaded with key fixing. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                else:
                    print("Model doesn't have load_state_dict method")
                    return False
            
            # Freeze early layers if requested
            if freeze_layers:
                # For ResNet models, freeze all layers except final FC
                if isinstance(self.model, (torch.nn.Module)):
                    for name, param in self.model.named_parameters():
                        # Don't freeze FC/classifier layers
                        if 'fc' not in name and 'classifier' not in name:
                            param.requires_grad = False
                        else:
                            print(f"Keeping {name} trainable")
                            
                print("Early layers frozen for transfer learning")
            
            # Modify specific adaptation layers if needed
            if adaptation_layers:
                # Example: add dropout or modify specific layers
                for layer_name in adaptation_layers:
                    if hasattr(self.model, layer_name):
                        layer = getattr(self.model, layer_name)
                        if layer_name == 'fc' and isinstance(layer, torch.nn.Linear):
                            # Add dropout before FC layer
                            in_features = layer.in_features
                            out_features = layer.out_features
                            dropout_layer = torch.nn.Dropout(0.5)
                            new_fc = torch.nn.Sequential(
                                dropout_layer,
                                torch.nn.Linear(in_features, out_features)
                            )
                            setattr(self.model, layer_name, new_fc)
                            print(f"Added dropout to {layer_name}")
                            
            # Reset optimizer with the new parameter settings
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config['learning_rate']
            )
            
            return True
        
        except Exception as e:
            print(f"Error adapting pretrained model: {str(e)}")
            return False

class CheckpointManager:
    """Manages model checkpointing and state restoration"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state: dict, is_best: bool = False, filename: str = None):
        """
        Save checkpoint with complete state
        Args:
            state: Dictionary containing state to save
            is_best: If True, also save as best model
            filename: Optional specific filename
        """
        if filename is None:
            filename = f'checkpoint_ep{state["episode"]}.pt'
            
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(state, filepath)
        
        # If best model, create a copy
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, 'model_best.pt')
            shutil.copyfile(filepath, best_filepath)
            
        return filepath
        
    def load_checkpoint(self, model, optimizer=None, scheduler=None, filename=None):
        """
        Load checkpoint and restore state
        Args:
            model: Model to restore
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            filename: Specific checkpoint file to load (if None, load latest)
        Returns:
            Loaded state dict and epoch number
        """
        if filename is None:
            # Find latest checkpoint
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_ep*.pt')))
            if not checkpoints:
                return None
            filename = checkpoints[-1]
            
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            
        # Restore model state
        model.load_state_dict(checkpoint['model_state'])
        
        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        # Restore scheduler state if provided
        if scheduler is not None and 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        return checkpoint
        
    def get_best_model_path(self):
        """Get path to best model checkpoint"""
        return os.path.join(self.checkpoint_dir, 'model_best.pt')

class AutomatedTrainingManager:
    def __init__(self, al_manager):
        self.al_manager = al_manager
        self.is_training = False
        self.stop_requested = False
        self.current_batch_labeled_count = 0
        self.current_batch_size = 0
        self.batch_complete = asyncio.Event()
        self.current_batch = []
        self.last_training_start = None
        self.training_timeout = 300  # 5 minutes timeout
        
        # Ensure training_config is properly initialized with default values
        self.training_config = {
            'sampling_strategy': 'least_confidence',  # default strategy
            'batch_size': 32,  # default batch size
            'epochs': 10,  # default epochs
            'learning_rate': 0.001  # default learning rate
        }
        
        self.min_required_samples = 10

    def update_config(self, config):
        """Update training configuration safely"""
        if not isinstance(config, dict):
            print(f"Warning: config is not a dictionary: {type(config)}")
            return
            
        # Update each field individually with type checking
        if 'sampling_strategy' in config and isinstance(config['sampling_strategy'], str):
            self.training_config['sampling_strategy'] = config['sampling_strategy']
            
        if 'batch_size' in config:
            try:
                batch_size = int(config['batch_size'])
                if batch_size > 0:
                    self.training_config['batch_size'] = batch_size
            except (ValueError, TypeError):
                print(f"Invalid batch_size: {config['batch_size']}")
                
        if 'epochs' in config:
            try:
                epochs = int(config['epochs'])
                if epochs > 0:
                    self.training_config['epochs'] = epochs
            except (ValueError, TypeError):
                print(f"Invalid epochs: {config['epochs']}")
                
        if 'learning_rate' in config:
            try:
                lr = float(config['learning_rate'])
                if lr > 0:
                    self.training_config['learning_rate'] = lr
            except (ValueError, TypeError):
                print(f"Invalid learning_rate: {config['learning_rate']}")
                
        print(f"Updated training config: {self.training_config}")
    
    def check_training_state(self):
        """Check if training state is stuck and reset if necessary"""
        if self.is_training and self.last_training_start:
            time_elapsed = time.time() - self.last_training_start
            if time_elapsed > self.training_timeout:
                print(f"Training state was stuck for {time_elapsed:.0f} seconds. Resetting...")
                self.is_training = False
                self.last_training_start = None
                return True
        return False

    def on_label_submitted(self):
        """Enhanced label submission handling"""
        self.current_batch_labeled_count += 1
        print(f"\n=== Automated Training Status ===")
        print(f"Labels in batch: {self.current_batch_labeled_count}/{self.current_batch_size}")
        print(f"Total labeled samples: {len(self.al_manager.labeled_data)}")
        
        # Check and reset if training is stuck
        self.check_training_state()
        
        # Only trigger training if we have completed the batch AND aren't already training
        if (self.current_batch_labeled_count >= self.current_batch_size and 
            len(self.al_manager.labeled_data) >= self.min_required_samples and
            not self.is_training):
            print("Starting training cycle...")
            asyncio.create_task(self._train_and_get_next_batch())
        else:
            if self.is_training:
                print("Training already in progress...")
            elif self.current_batch_labeled_count < self.current_batch_size:
                print(f"Waiting for more labels before training")
            elif len(self.al_manager.labeled_data) < self.min_required_samples:
                print(f"Need more samples (have {len(self.al_manager.labeled_data)}, need {self.min_required_samples})")
            
    def update_config(self, config):
        """Update training configuration"""
        self.training_config.update(config)

    async def start_automated_training(self, config: dict):
        """Start automated training with improved state management"""
        try:
            print(f"Received config: {config}")
            
            # Check and reset if training is stuck
            if self.check_training_state():
                print("Reset stuck training state")
            
            if self.is_training:
                return {"status": "already_running"}
                
            self.training_config = {
                'epochs': int(config['epochs']),
                'batch_size': int(config['batch_size']),
                'sampling_strategy': str(config['sampling_strategy']),
                'learning_rate': float(config.get('learning_rate', 0.001))
            }
            
            print(f"Processed config: {self.training_config}")
            
            self.is_training = True
            self.stop_requested = False
            self.last_training_start = time.time()
            
            # Get first batch if needed
            if not self.current_batch:
                self.current_batch = self.al_manager.get_next_batch(
                    strategy=self.training_config['sampling_strategy'],
                    batch_size=self.training_config['batch_size']
                )
                self.current_batch_size = len(self.current_batch)
                self.current_batch_labeled_count = 0
                
            return {
                "status": "success",
                "message": "Started automated training",
                "batch_size": self.current_batch_size,
                "config": self.training_config
            }
                
        except Exception as e:
            self.is_training = False
            self.last_training_start = None
            print(f"Error in start_automated_training: {str(e)}")
            raise

    async def _train_and_get_next_batch(self):
        """Training cycle with improved state management"""
        try:
            print("\n=== Starting Training Cycle ===")
            self.last_training_start = time.time()
            
            if len(self.al_manager.labeled_data) < self.min_required_samples:
                print(f"Insufficient labeled data ({len(self.al_manager.labeled_data)} < {self.min_required_samples})")
                self.is_training = False
                return
            
            # Train the model
            training_result = self.al_manager.train_episode(
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                learning_rate=self.training_config['learning_rate']
            )
            
            print(f"Training completed. Validation accuracy: {self.al_manager.best_val_acc:.2f}%")
            
            if not self.stop_requested:
                print("Getting next batch...")
                self.current_batch = self.al_manager.get_next_batch(
                    strategy=self.training_config['sampling_strategy'],
                    batch_size=self.training_config['batch_size']
                )
                self.current_batch_size = len(self.current_batch)
                self.current_batch_labeled_count = 0
                
                self.al_manager.current_batch = [x["image_id"] for x in self.current_batch]
                self.is_training = False
                self.last_training_start = None
                
                return {
                    "status": "success",
                    "new_batch_available": True,
                    "batch_size": self.current_batch_size,
                    "training_result": training_result,
                    "validation_accuracy": self.al_manager.best_val_acc
                }
                    
        except Exception as e:
            print(f"Training error: {str(e)}")
            self.is_training = False
            self.last_training_start = None
            raise
    
    async def get_new_batch(self):
        """Manually request a new batch"""
        if not self.is_training:
            raise HTTPException(status_code=400, detail="Automated training not active")
        
        try:
            self.current_batch = self.al_manager.get_next_batch(
                strategy=self.training_config['sampling_strategy'],
                batch_size=self.training_config['batch_size']
            )
            self.current_batch_size = len(self.current_batch)
            self.current_batch_labeled_count = 0
            self.batch_complete.clear()
            
            return {
                "status": "success",
                "batch_size": self.current_batch_size
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def stop_automated_training(self):
        """Stop the automated training cycle"""
        self.stop_requested = True
        self.batch_complete.set()  # Release any waiting
            
    def get_training_status(self):
        """Get current automated training status"""
        return {
            "is_training": self.is_training,
            "current_episode": self.al_manager.episode,
            "labeled_count": len(self.al_manager.labeled_data),
            "unlabeled_count": len(self.al_manager.unlabeled_data),
            "current_batch": {
                "labeled": self.current_batch_labeled_count,
                "total": self.current_batch_size
            },
            "config": {
                "sampling_strategy": self.al_manager.config.get('sampling_strategy'),
                "batch_size": self.al_manager.config.get('batch_size'),
                "epochs": self.al_manager.config.get('epochs'),
                "learning_rate": self.al_manager.config.get('learning_rate')
            }
        }

    async def train_current_model(self):
        """Train the current model and collect metrics"""
        training_result = self.al_manager.train_episode(
            epochs=self.al_manager.config.get('epochs', 10),
            batch_size=self.al_manager.config.get('batch_size', 32),
            learning_rate=self.al_manager.config.get('learning_rate', 0.001)
        )
        
        # Collect metrics after training
        metrics = {
            'training_metrics': training_result,
            'episode_accuracies': {
                'x': self.al_manager.plot_episode_xvalues,
                'y': self.al_manager.plot_episode_yvalues
            },
            'current_epoch_losses': {
                'x': self.al_manager.plot_epoch_xvalues,
                'y': self.al_manager.plot_epoch_yvalues
            },
            'validation_accuracy': self.al_manager.best_val_acc,
            'episode': self.al_manager.episode
        }
        
        # Store metrics for UI to access
        self.current_metrics = metrics
        return metrics

class LRSchedulerManager:
    """Manages different learning rate scheduling strategies"""
    
    def __init__(self, optimizer, strategy="plateau", **kwargs):
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = kwargs.get('initial_lr', 0.001)
        self.history = []
        self.scheduler = self._create_scheduler(**kwargs)
        
    def get_lr(self):
        """Get current learning rate"""
        try:
            return self.optimizer.param_groups[0]['lr']
        except Exception as e:
            print(f"Error getting learning rate: {str(e)}")
            return self.initial_lr
    
    def get_status(self):
        """Get current scheduler status"""
        return {
            "strategy": self.strategy,
            "current_lr": self.get_lr(),
            "history": self.history,
            "initial_lr": self.initial_lr
        }
        
    def _create_scheduler(self, **kwargs):
        """Create scheduler based on strategy"""
        if self.strategy == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 5),
                verbose=kwargs.get('verbose', True),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif self.strategy == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('min_lr', 0)
            )
        elif self.strategy == "warmup":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get('max_lr', 0.1),
                epochs=kwargs.get('epochs', 30),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100),
                pct_start=kwargs.get('warmup_pct', 0.3)
            )
        elif self.strategy == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler strategy: {self.strategy}")
            
    def step(self, metric=None):
        """Update learning rate based on metric or epoch"""
        current_lr = self.get_lr()
        
        if self.strategy == "plateau":
            if metric is None:
                raise ValueError("Metric required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
        new_lr = self.get_lr()
        
        # Record history
        self.history.append({
            'old_lr': current_lr,
            'new_lr': new_lr,
            'metric': metric
        })
        
        return new_lr
        
    def reset(self):
        """Reset learning rate to initial value"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
            
    def state_dict(self):
        """Get scheduler state for checkpointing"""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'strategy': self.strategy,
            'history': self.history
        }
        
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.strategy = state_dict['strategy']
        self.history = state_dict['history']

# FastAPI app setup
app = FastAPI()
al_manager = ActiveLearningManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def adapt_pretrained_model(self, model_state, freeze_layers=True, adaptation_layers=None):
    """
    Adapt a pre-trained model for active learning by optionally freezing layers
    and preparing the model for fine-tuning.
    
    Args:
        model_state: State dict of the pretrained model
        freeze_layers: Whether to freeze early layers
        adaptation_layers: List of layer names to specifically adapt
    """
    try:
        # Load the pretrained weights
        if hasattr(self.model, 'load_state_dict'):
            # Try to load directly, with a non-strict option to allow for differences
            try:
                self.model.load_state_dict(model_state, strict=False)
                print("Loaded pretrained model weights (non-strict)")
            except Exception as e:
                print(f"Error loading model directly: {str(e)}")
                
                # Try to fix common key mismatches
                fixed_state_dict = {}
                for k, v in model_state.items():
                    # Handle module prefix differences (common with DataParallel)
                    if k.startswith('module.') and not any(key.startswith('module.') for key in self.model.state_dict()):
                        fixed_state_dict[k[7:]] = v
                    elif not k.startswith('module.') and any(key.startswith('module.') for key in self.model.state_dict()):
                        fixed_state_dict['module.' + k] = v
                    else:
                        fixed_state_dict[k] = v
                
                # Try loading with fixed keys
                missing_keys, unexpected_keys = self.model.load_state_dict(fixed_state_dict, strict=False)
                print(f"Loaded with key fixing. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        else:
            print("Model doesn't have load_state_dict method")
            return False
        
        # Freeze early layers if requested
        if freeze_layers:
            # For ResNet models, freeze all layers except final FC
            if isinstance(self.model, (torch.nn.Module)):
                for name, param in self.model.named_parameters():
                    # Don't freeze FC/classifier layers
                    if 'fc' not in name and 'classifier' not in name:
                        param.requires_grad = False
                    else:
                        print(f"Keeping {name} trainable")
                        
            print("Early layers frozen for transfer learning")
        
        # Modify specific adaptation layers if needed
        if adaptation_layers:
            # Example: add dropout or modify specific layers
            for layer_name in adaptation_layers:
                if hasattr(self.model, layer_name):
                    layer = getattr(self.model, layer_name)
                    if layer_name == 'fc' and isinstance(layer, torch.nn.Linear):
                        # Add dropout before FC layer
                        in_features = layer.in_features
                        out_features = layer.out_features
                        dropout_layer = torch.nn.Dropout(0.5)
                        new_fc = torch.nn.Sequential(
                            dropout_layer,
                            torch.nn.Linear(in_features, out_features)
                        )
                        setattr(self.model, layer_name, new_fc)
                        print(f"Added dropout to {layer_name}")
        
        return True
    
    except Exception as e:
        print(f"Error adapting pretrained model: {str(e)}")
        return False

# Add a new endpoint to perform model adaptation
@app.post("/adapt-pretrained-model")
async def adapt_pretrained_model(
    freeze_layers: bool = Form(True),
    adaptation_type: str = Form("full_finetune")
):
    """
    Adapt a previously imported model for active learning
    """
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model has been imported yet")
        
        # Get the current model state
        model_state = al_manager.model.state_dict()
        
        # Determine adaptation layers based on adaptation type
        adaptation_layers = None
        if adaptation_type == "last_layer":
            adaptation_layers = ["fc"]  # Only adapt final layer
        elif adaptation_type == "mid_layers":
            adaptation_layers = ["layer4", "fc"]  # Adapt last conv block and FC
        
        # Perform adaptation
        success = al_manager.adapt_pretrained_model(
            model_state=model_state,
            freeze_layers=freeze_layers,
            adaptation_layers=adaptation_layers
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to adapt model")
        
        return {
            "status": "success",
            "message": "Model adapted successfully",
            "adaptation_type": adaptation_type,
            "freeze_layers": freeze_layers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")

@app.post("/init")
async def initialize_project(request: ProjectInit):
    """Initialize new active learning project with improved error handling"""
    try:
        print(f"Initializing project with config: {request}")
        
        # Create training config from request
        training_config = {
            'sampling_strategy': request.sampling_strategy if hasattr(request, 'sampling_strategy') else 'least_confidence',
            'batch_size': request.batch_size if hasattr(request, 'batch_size') else 32,
            'epochs': request.epochs if hasattr(request, 'epochs') else 10,
            'learning_rate': request.learning_rate if hasattr(request, 'learning_rate') else 0.001,
        }
        
        # Initialize project with config
        result = al_manager.initialize_project(
            project_name=request.project_name,
            model_name=request.model_type,
            num_classes=request.num_classes,
            config=training_config
        )
        
        return {
            "status": "success",
            "output_dir": result["output_dir"],
            "config": training_config
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"Error initializing project: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_initial_data(
    files: List[UploadFile],
    val_split: float = None,
    initial_labeled_ratio: float = None
):
    """Upload and split initial dataset"""
    try:
        if val_split is not None:
            al_manager.config['val_split'] = val_split
        if initial_labeled_ratio is not None:
            al_manager.config['initial_labeled_ratio'] = initial_labeled_ratio
            
        return await al_manager.add_initial_data(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-batch")
async def get_batch(request: BatchRequest):
    try:
        if not hasattr(al_manager, 'model') or al_manager.model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not initialized. Please initialize project first."
            )
            
        if len(al_manager.unlabeled_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No unlabeled data available"
            )
            
        return al_manager.get_next_batch(
            strategy=request.strategy,
            batch_size=request.batch_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-label")
async def submit_label(submission: LabelSubmission):
    """Submit label for an image with enhanced logging"""
    try:
        print(f"\n=== Label Submission ===")
        print(f"Image ID: {submission.image_id}, Label: {submission.label}")
        
        result = al_manager.submit_label(
            image_id=submission.image_id,
            label=submission.label
        )
        
        # Notify automated trainer of label submission
        if automated_trainer.is_training:
            print("Automated trainer is already training")
        else:
            print(f"Current batch progress: {automated_trainer.current_batch_labeled_count + 1}/{automated_trainer.current_batch_size}")
            automated_trainer.on_label_submitted()
            
            # Check if this submission completes the batch
            batch_complete = automated_trainer.current_batch_labeled_count >= automated_trainer.current_batch_size
            if batch_complete:
                print("Batch complete - Initiating training cycle")
            
        return {
            **result,  # Include the original result
            "batch_complete": automated_trainer.current_batch_labeled_count >= automated_trainer.current_batch_size,
            "is_training": automated_trainer.is_training,
            "current_progress": {
                "labeled": automated_trainer.current_batch_labeled_count,
                "total": automated_trainer.current_batch_size
            }
        }
    except Exception as e:
        print(f"Error in submit_label: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train/retrain model on labeled data"""
    try:
        if not al_manager.model:
            raise HTTPException(
                status_code=400, 
                detail="Model not initialized. Please initialize project first."
            )
            
        if len(al_manager.labeled_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No labeled data available for training"
            )
            
        return al_manager.train(epochs, batch_size, learning_rate)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current status of active learning process"""
    return {
        "project_name": al_manager.project_name,
        "current_episode": al_manager.episode,
        "labeled_count": len(al_manager.labeled_data),
        "unlabeled_count": len(al_manager.unlabeled_data),
        "validation_count": len(al_manager.validation_data),
        "current_batch_size": len(al_manager.current_batch),
    }

@app.get("/metrics")
async def get_metrics():
    try:
        metrics = {
            "best_val_acc": al_manager.best_val_acc,
            "current_episode": al_manager.episode,
            "episode_accuracies": {
                "x": al_manager.plot_episode_xvalues,
                "y": al_manager.plot_episode_yvalues
            },
            "current_epoch_losses": {
                "x": al_manager.plot_epoch_xvalues,
                "y": al_manager.plot_epoch_yvalues
            }
        }
        print("Sending metrics:", metrics)  # Debug log
        return metrics
    except Exception as e:
        print(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{image_id}")
async def get_image(image_id: int):
    """Get image data for display with improved error handling and debugging"""
    try:
        print(f"Serving image {image_id}")
        
        # Check if the ID is valid
        if not isinstance(image_id, int) or image_id < 0:
            print(f"Invalid image ID format: {image_id}")
            raise HTTPException(status_code=400, detail="Invalid image ID format")
        
        # Get tensor based on where the image is stored
        tensor = None
        location = None
        
        if image_id in al_manager.unlabeled_data:
            tensor = al_manager.unlabeled_data[image_id]
            location = "unlabeled_data"
        elif image_id in al_manager.labeled_data:
            tensor = al_manager.labeled_data[image_id][0]
            location = "labeled_data"
        elif image_id in al_manager.validation_data:
            tensor = al_manager.validation_data[image_id][0]
            location = "validation_data"
        
        if tensor is None:
            print(f"Image {image_id} not found in any dataset")
            print(f"Available IDs in unlabeled: {list(al_manager.unlabeled_data.keys())[:5]}...")
            print(f"Available IDs in labeled: {list(al_manager.labeled_data.keys())[:5]}...")
            print(f"Available IDs in validation: {list(al_manager.validation_data.keys())[:5]}...")
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found in any dataset")
            
        print(f"Found image {image_id} in {location}")
            
        # Ensure tensor is on CPU and in the right format
        try:
            tensor = tensor.cpu()
        except Exception as e:
            print(f"Error moving tensor to CPU: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image tensor")
        
        # Check tensor shape and type
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        try:
            # Denormalize the tensor
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
                
            # Convert tensor to image
            img_array = tensor.numpy().transpose(1, 2, 0)
            img_array = np.clip(img_array, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            
            # Check array shape and values
            print(f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
            
            img = Image.fromarray(img_array)
            
            # Save to byte stream with error handling
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Add caching headers
            headers = {
                'Cache-Control': 'public, max-age=31536000',
                'ETag': f'"{hash(img_byte_arr)}"'
            }
            
            return Response(
                content=img_byte_arr, 
                media_type="image/png",
                headers=headers
            )
        except Exception as e:
            error_msg = f"Error converting tensor to image: {str(e)}"
            print(error_msg)
            traceback.print_exc()  # Print full traceback
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        error_msg = f"Error serving image {image_id}: {str(e)}"
        print(error_msg)
        traceback.print_exc()  # Print full traceback
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/export-model")
async def export_model():
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
        
        # Get current configuration
        config = {
            'sampling_strategy': automated_trainer.training_config['sampling_strategy'],
            'epochs': automated_trainer.training_config['epochs'],
            'batch_size': automated_trainer.training_config['batch_size'],
            'learning_rate': automated_trainer.training_config['learning_rate']
        }
        
        model_export = {
            'model_state': al_manager.model.state_dict(),
            'model_config': {
                'project_name': al_manager.project_name,
                'episode': al_manager.episode,
                'model_name': al_manager.model.__class__.__name__,
                'training_config': config,  # Save full training config
                'metrics': {
                    'episode_accuracies': {
                        'x': al_manager.plot_episode_xvalues,
                        'y': al_manager.plot_episode_yvalues
                    }
                }
            }
        }
        
        filename = f"{al_manager.project_name}_{config['sampling_strategy']}_e{config['epochs']}_b{config['batch_size']}_model.pt"
        export_path = os.path.join(al_manager.output_dir, filename)
        torch.save(model_export, export_path)
        
        return FileResponse(
            path=export_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/import-model")
async def import_model(uploaded_file: UploadFile = File(...)):  # Add File import and make it required
    """Import a previously exported model"""
    try:
        # Read model file
        content = await uploaded_file.read()
        
        # Load model data with safe device mapping
        if torch.cuda.is_available():
            model_data = torch.load(io.BytesIO(content))
        else:
            model_data = torch.load(io.BytesIO(content), map_location=torch.device('cpu'))
            
        # Load configuration
        config = model_data['model_config']
        training_config = config.get('training_config', {})
        
        # Initialize model first if not already done
        if al_manager.model is None:
            num_classes = len(config.get('labels', [])) or config.get('num_classes', 2)
            init_result = await al_manager.initialize_project(
                project_name=config['project_name'],
                model_name=config.get('model_type', 'resnet50'),
                num_classes=num_classes
            )
            
        # Set all the manager properties from the imported model
        al_manager.project_name = config['project_name']
        al_manager.episode = config['episode']
        
        # Safely set metrics
        metrics = config.get('metrics', {})
        episode_accuracies = metrics.get('episode_accuracies', {'x': [], 'y': []})
        al_manager.plot_episode_xvalues = episode_accuracies['x']
        al_manager.plot_episode_yvalues = episode_accuracies['y']
        
        # Load model state
        al_manager.model.load_state_dict(model_data['model_state'])
        
        # Update automated trainer config if available
        if 'automated_trainer' in globals():
            automated_trainer.training_config.update({
                'sampling_strategy': training_config.get('sampling_strategy', 'least_confidence'),
                'epochs': training_config.get('epochs', 10),
                'batch_size': training_config.get('batch_size', 32),
                'learning_rate': training_config.get('learning_rate', 0.001)
            })
        
        return {
            "status": "success",
            "project_name": al_manager.project_name,
            "episode": al_manager.episode,
            "training_config": training_config,
            "metrics": {
                "episode_accuracies": {
                    "x": al_manager.plot_episode_xvalues,
                    "y": al_manager.plot_episode_yvalues
                }
            }
        }
        
    except Exception as e:
        print(f"Error importing model: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/train-episode")
async def train_episode(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    try:
        if not hasattr(al_manager, 'model') or al_manager.model is None:
            raise HTTPException(
                status_code=400, 
                detail="Model not initialized. Please initialize project first."
            )
            
        if len(al_manager.labeled_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No labeled data available for training"
            )
            
        result = al_manager.train_episode(epochs, batch_size, learning_rate)
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Training episode error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/episode-history")
async def get_episode_history():
    return {
        "episodes": al_manager.episode_history,
        "current_episode": al_manager.episode
    }

@app.get("/validation-status")
async def get_validation_status():
    """Get validation set labeling status"""
    return al_manager.get_validation_status()

automated_trainer = AutomatedTrainingManager(al_manager)

# Add new endpoints
@app.post("/start-automated-training")
async def start_automated_training(config: TrainingConfig):
    """Start automated training with configuration"""
    try:
        await automated_trainer.start_automated_training(config.dict())
        return {"status": "success", "message": "Automated training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-automated-training")
async def stop_automated_training():
    """Stop automated active learning cycle"""
    try:
        automated_trainer.stop_automated_training()
        return {"status": "success", "message": "Automated training stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add automated training status endpoint
@app.get("/automated-training-status")
async def get_automated_training_status():
    """Get current automated training status"""
    return automated_trainer.get_training_status()

@app.post("/get-next-batch")
async def get_next_batch():
    """Manually get next batch during automated training with improved error handling"""
    try:
        print("\n=== Getting Next Batch ===")
        print(f"Current state: Labeled data: {len(al_manager.labeled_data)}, Unlabeled data: {len(al_manager.unlabeled_data)}")
        
        # Make sure we have unlabeled data
        if len(al_manager.unlabeled_data) == 0:
            print("No unlabeled data available")
            return {
                "status": "error",
                "error": "No unlabeled data available for batch selection",
                "unlabeled_count": 0,
                "labeled_count": len(al_manager.labeled_data),
                "validation_count": len(al_manager.validation_data)
            }
        
        # Make sure batch size is valid, with explicit type checking and fallbacks
        batch_size = 32  # Default batch size
        try:
            config_batch_size = automated_trainer.training_config.get('batch_size')
            if config_batch_size is not None and isinstance(config_batch_size, (int, float)) and config_batch_size > 0:
                batch_size = int(config_batch_size)
        except (AttributeError, TypeError) as e:
            print(f"Error getting batch size from config: {str(e)}")
            print(f"Using default batch size: {batch_size}")
        
        # Add safety check for batch size
        if batch_size > len(al_manager.unlabeled_data):
            print(f"Batch size {batch_size} is larger than available unlabeled data {len(al_manager.unlabeled_data)}")
            # Adjust batch size automatically
            batch_size = len(al_manager.unlabeled_data)
            print(f"Adjusted batch size to {batch_size}")
        
        # Get strategy with fallbacks
        strategy = "least_confidence"  # Default strategy
        try:
            config_strategy = automated_trainer.training_config.get('sampling_strategy')
            if config_strategy is not None and isinstance(config_strategy, str):
                strategy = config_strategy
        except (AttributeError, TypeError) as e:
            print(f"Error getting strategy from config: {str(e)}")
            print(f"Using default strategy: {strategy}")
            
        print(f"Getting batch using strategy: {strategy}, batch size: {batch_size}")
        
        try:
            batch = al_manager.get_next_batch(strategy=strategy, batch_size=batch_size)
            
            # Store batch information
            try:
                automated_trainer.current_batch = batch
                automated_trainer.current_batch_size = len(batch)
                automated_trainer.current_batch_labeled_count = 0
            except Exception as config_error:
                print(f"Error updating trainer state: {str(config_error)}")
                # Continue even if we can't update the automated trainer state
            
            print(f"Successfully got batch of {len(batch)} images")
            
            return {
                "status": "success",
                "batch_size": len(batch),
                "strategy": strategy
            }
        except Exception as e:
            print(f"Error getting batch: {str(e)}")
            traceback.print_exc()  # Print full stack trace
            
            # Try again with random strategy as fallback
            try:
                print("Trying fallback random sampling strategy")
                batch = al_manager.get_next_batch(strategy="random", batch_size=batch_size)
                
                # Store batch information
                try:
                    automated_trainer.current_batch = batch
                    automated_trainer.current_batch_size = len(batch)
                    automated_trainer.current_batch_labeled_count = 0
                except Exception as config_error:
                    print(f"Error updating trainer state: {str(config_error)}")
                
                print(f"Successfully got batch using fallback strategy with {len(batch)} images")
                
                return {
                    "status": "success",
                    "batch_size": len(batch),
                    "strategy": "random (fallback)"
                }
            except Exception as fallback_error:
                print(f"Fallback strategy also failed: {str(fallback_error)}")
                traceback.print_exc()
                raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-training-state")
async def reset_training_state():
    """Reset stuck training state"""
    if automated_trainer.check_training_state():
        return {"status": "success", "message": "Reset stuck training state"}
    return {"status": "success", "message": "Training state is not stuck"}

@app.post("/save-checkpoint")
async def save_checkpoint():
    """Save current model checkpoint"""
    try:
        checkpoint_path = al_manager.save_state()
        return {
            "status": "success",
            "checkpoint_path": checkpoint_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-checkpoint")
async def load_checkpoint(checkpoint_path: str = None):
    """Load model checkpoint"""
    try:
        checkpoint = al_manager.load_state(checkpoint_path)
        if checkpoint:
            return {
                "status": "success",
                "episode": checkpoint['episode'],
                "best_val_acc": checkpoint['best_val_acc']
            }
        else:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    try:
        checkpoints = glob.glob(os.path.join(al_manager.output_dir, 
            'checkpoints', 'checkpoint_ep*.pt'))
        return {
            "checkpoints": [os.path.basename(cp) for cp in checkpoints]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure-lr-scheduler")
async def configure_lr_scheduler(config: dict):
    """Configure learning rate scheduler"""
    try:
        al_manager.lr_config.update(config)
        return {"status": "success", "config": al_manager.lr_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lr-scheduler-status")
async def get_lr_scheduler_status():
    """Get current learning rate scheduler status"""
    try:
        if al_manager.lr_scheduler is None:
            return {
                "strategy": "plateau",  # default strategy
                "current_lr": 0.001,    # default learning rate
                "history": [],
                "initial_lr": 0.001
            }
            
        return al_manager.lr_scheduler.get_status()
    except Exception as e:
        print(f"Error getting LR scheduler status: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get scheduler status: {str(e)}"
        )

@app.post("/import-pretrained-model")
async def import_pretrained_model(
    uploaded_file: UploadFile = File(...),
    model_type: str = Form(...),
    num_classes: int = Form(2),
    project_name: str = Form("imported_project")
):
    """
    Import a pre-trained model that wasn't created with this UI
    """
    try:
        # Read model file
        content = await uploaded_file.read()
        
        # Validate model type - now allowing additional model types
        supported_models = ["resnet18", "resnet50", "efficientnet", "densenet", "mobilenet", "custom"]
        
        if model_type not in supported_models:
            # If not in our supported list, automatically assign to "custom"
            print(f"Model type '{model_type}' not in supported list, treating as 'custom'")
            model_type = "custom"
        
        # Load model state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Initialize model architecture based on type
        if model_type == "custom":
            # For custom models, we'll need to initialize one of our standard architectures 
            # and then adapt it to match the imported model's structure
            fallback_model_type = "resnet50"  # Use this as fallback
            
            # Initialize project with fallback model
            if not al_manager.project_name:
                init_result = al_manager.initialize_project(
                    project_name=project_name,
                    model_name=fallback_model_type,
                    num_classes=num_classes
                )
        else:
            # Initialize project with specified model
            if not al_manager.project_name:
                init_result = al_manager.initialize_project(
                    project_name=project_name,
                    model_name=model_type,
                    num_classes=num_classes
                )
        
        # Try to load the model using our safe utility
        state_dict = safe_load_model(tmp_path)
        
        if state_dict is None:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
            raise ValueError("Failed to load model file with any method. The file may be corrupted or in an unsupported format.")
            
        # Extract the model state based on the structure
        model_state = extract_model_state(state_dict)
        
        # Handle custom model structure (architecture adaptation)
        if model_type == "custom":
            # Try to detect the number of classes from the model state
            try:
                detected_classes = detect_num_classes(model_state)
                if detected_classes and detected_classes != num_classes:
                    print(f"Detected {detected_classes} classes in model, updating from {num_classes}")
                    num_classes = detected_classes
                    
                    # Reinitialize the model with detected class count
                    init_result = al_manager.initialize_project(
                        project_name=project_name,
                        model_name=fallback_model_type,
                        num_classes=num_classes
                    )
            except Exception as e:
                print(f"Could not detect classes: {str(e)}")
        
        # Attempt to load the state dict with flexible matching
        try:
            # First try direct loading
            al_manager.model.load_state_dict(model_state, strict=False)
            print("Loaded model state with non-strict matching")
        except Exception as e:
            print(f"Direct loading error: {str(e)}")
            
            # Try key remapping for common patterns
            fixed_state_dict = {}
            for k, v in model_state.items():
                # Remove 'module.' prefix (from DataParallel models)
                if k.startswith('module.'):
                    fixed_state_dict[k[7:]] = v
                # Add 'module.' prefix if needed
                elif not k.startswith('module.') and f'module.{k}' in al_manager.model.state_dict():
                    fixed_state_dict[f'module.{k}'] = v
                else:
                    fixed_state_dict[k] = v
            
            # Attempt to load with fixed keys
            missing_keys, unexpected_keys = al_manager.model.load_state_dict(fixed_state_dict, strict=False)
            print(f"Loaded with key fixing. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "message": "Pre-trained model imported successfully",
            "project_name": project_name,
            "model_type": model_type,
            "num_classes": num_classes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@app.post("/verify-model-compatibility")
async def verify_model_compatibility(uploaded_file: UploadFile = File(...)):
    """
    Check if a model file is compatible with the system before fully importing it
    """
    try:
        # Read model file
        content = await uploaded_file.read()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Try to load the model using our safe utility
            state_dict = safe_load_model(tmp_path)
            
            if state_dict is None:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
                return {
                    "status": "error",
                    "compatible": False,
                    "message": "Unable to load model file with any method. The file may be corrupted or in an unsupported format."
                }
            
            # Analyze the model structure
            model_info = analyze_model_structure(state_dict)
            print(f"Model info: {model_info}")
            
            # Clean up
            os.unlink(tmp_path)
            
            return {
                "status": "success",
                "compatible": model_info["compatible"],
                "model_type": model_info["detected_type"],
                "num_classes": model_info["num_classes"],
                "adaptation_needed": model_info["adaptation_needed"],
                "message": model_info["message"]
            }
            
        except Exception as e:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
            return {
                "status": "error",
                "compatible": False,
                "message": f"Error analyzing model file: {str(e)}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    
def extract_model_state(state_dict):
    """
    Extract model state from various checkpoint formats
    Handles different model saving formats
    """
    if isinstance(state_dict, dict):
        # Our export format
        if 'model_state' in state_dict:
            return state_dict['model_state']
        # Common PyTorch checkpoint format
        elif 'state_dict' in state_dict:
            return state_dict['state_dict']
        # Direct state dict from weights_only=True loading
        elif any(k.endswith('.weight') or k.endswith('.bias') for k in state_dict.keys()):
            return state_dict
        # Models with 'model' key from HuggingFace or similar
        elif 'model' in state_dict and isinstance(state_dict['model'], dict):
            return state_dict['model']
        # Torchvision model zoo style
        elif any(k.startswith('layer') or k.startswith('conv') or k.startswith('fc') 
                or k.startswith('features') or k.startswith('classifier') for k in state_dict.keys()):
            return state_dict
    
    # If the loaded object is a model itself (unlikely but possible)
    if hasattr(state_dict, 'state_dict') and callable(getattr(state_dict, 'state_dict')):
        try:
            return state_dict.state_dict()
        except:
            pass
    
    # Default: return as-is and hope for the best
    return state_dict

def analyze_model_structure(state_dict):
    """
    Analyze the model structure to determine compatibility and required adaptations
    Handles various model formats and provides detailed information
    """
    result = {
        "compatible": False,
        "detected_type": "unknown",
        "num_classes": None,
        "adaptation_needed": True,
        "message": ""
    }
    
    try:
        # Extract model state
        model_state = extract_model_state(state_dict)
        
        # Check if it's an empty dictionary
        if not model_state or not isinstance(model_state, dict):
            result["message"] = "Unable to extract model state dictionary. Invalid format."
            return result
            
        # Print some keys for debugging
        print(f"Keys in model state: {list(model_state.keys())[:5]}...")
        
        # Check model type by examining keys
        key_set = set([k.split('.')[0] for k in model_state.keys()])
        
        # Detect Vision Transformer (ViT) models
        if any(k in ['cls_token', 'pos_embed', 'patch_embed'] for k in key_set):
            result["detected_type"] = "vision-transformer"
        # Detect ResNet models
        elif any(k.startswith('layer') for k in model_state.keys()):
            result["detected_type"] = "resnet"
        # Detect VGG-style models
        elif 'features' in key_set and 'classifier' in key_set:
            result["detected_type"] = "vgg-style"
        # Detect MobileNet models
        elif 'blocks' in key_set:
            result["detected_type"] = "mobilenet"
        # Detect BERT/transformer models
        elif any(k in ['encoder', 'decoder', 'transformer'] for k in key_set):
            result["detected_type"] = "transformer-nlp"
        # Detect detection models
        elif 'backbone' in key_set:
            result["detected_type"] = "detection-model"
        else:
            result["detected_type"] = "custom"
        
        # Try to detect number of classes
        num_classes = detect_num_classes(model_state)
        result["num_classes"] = num_classes
        
        # Determine compatibility and needs
        model_compatible = any(k.endswith('.weight') for k in model_state.keys())
        
        result["compatible"] = model_compatible
        result["adaptation_needed"] = True
        
        if model_compatible:
            result["message"] = f"Detected {result['detected_type']} architecture. "
            if num_classes:
                result["message"] += f"Found {num_classes} output classes. "
            result["message"] += "Will attempt adaptation through transfer learning."
        else:
            result["message"] = "Model format not recognized. Unable to determine compatibility."
        
        return result
    
    except Exception as e:
        result["message"] = f"Error analyzing model: {str(e)}"
        return result

def detect_num_classes(state_dict):
    """
    Try to detect number of classes from model state dict
    Handles different naming conventions for final layer
    """
    # Common patterns for output layer names in different architectures
    output_patterns = [
        # CNN models
        'fc.weight', 'classifier.weight', 'head.weight', 'output.weight',
        # For ViT and transformer models
        'head.weight', 'mlp_head.fc2.weight', 'cls_head.weight', 'classifier.weight',
        # For detection models
        'roi_heads.box_predictor.cls_score.weight', 'bbox_pred.weight'
    ]
    
    # Look for known output layer patterns
    for pattern in output_patterns:
        matching_keys = [k for k in state_dict.keys() if k.endswith(pattern)]
        if matching_keys:
            key = matching_keys[0]
            try:
                shape = state_dict[key].shape
                
                if len(shape) == 2:  # Linear layer weights are 2D [out_features, in_features]
                    out_features = shape[0]
                    return out_features
                elif len(shape) == 1:  # Sometimes weights can be flattened
                    return shape[0]
            except:
                continue
    
    # If no exact match, try a more flexible approach for the last layer
    try:
        # Find all weight parameters
        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
        
        # Look for likely classifier layers
        classifier_patterns = ['fc', 'classifier', 'head', 'output', 'linear', 'pred']
        for pattern in classifier_patterns:
            candidates = [k for k in weight_keys if pattern in k.lower()]
            if candidates:
                # Take the key with the highest index if it has numeric suffixes
                key = candidates[-1]  # Default to last one
                shape = state_dict[key].shape
                
                if len(shape) == 2:  # Linear layer
                    return shape[0]
                elif len(shape) == 1:  # Flattened weights
                    return shape[0]
    except Exception as e:
        print(f"Error in flexible class detection: {e}")
    
    # Special handling for ViT models
    if 'patch_embed.proj.weight' in state_dict:
        # Try to find other clues in the architecture
        try:
            # Check if this is a MAE model (self-supervised pre-training)
            if 'decoder_pred.weight' in state_dict:
                shape = state_dict['decoder_pred.weight'].shape
                return shape[0]
            
            # Check for full ViT with a classification head
            if 'head.weight' in state_dict:
                shape = state_dict['head.weight'].shape
                return shape[0]
        except:
            pass
    
    # If we can't determine, return None
    return None

@app.post("/upload-csv-paths")
async def upload_csv_paths(
    csv_file: UploadFile = File(...),
    delimiter: str = Form(","),
    val_split: float = Form(0.2),
    initial_labeled_ratio: float = Form(0.4)
):
    """
    Process a CSV file with image paths and load the images
    """
    try:
        # Read CSV content
        content = await csv_file.read()
        text_content = content.decode('utf-8', errors='replace')  # Handle encoding issues
        
        print(f"Using delimiter: '{delimiter}'")
        
        # Handle special characters
        if delimiter == "\\t" or delimiter == "tab":
            delimiter = "\t"
        
        # Parse CSV
        import csv
        from io import StringIO
        
        # Try to detect if our delimiter guess is wrong
        first_line = text_content.split('\n')[0]
        common_delimiters = [',', '\t', ';', '|']
        
        # If our delimiter doesn't appear in the first line but others do, we might have guessed wrong
        if delimiter not in first_line:
            for alt_delimiter in common_delimiters:
                if alt_delimiter in first_line:
                    print(f"Warning: Specified delimiter '{delimiter}' not found in first line. " +
                          f"Found '{alt_delimiter}' instead. Trying that...")
                    delimiter = alt_delimiter
                    break
        
        try:
            csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
            fieldnames = csv_reader.fieldnames
            
            # Check for file_path column
            if not fieldnames or 'file_path' not in fieldnames:
                # Try other common column names
                possible_column_names = ['file_path', 'filepath', 'path', 'filename', 'image', 'imagepath', 'file', 'image_path']
                
                found_column = None
                for column in possible_column_names:
                    if fieldnames and column in fieldnames:
                        found_column = column
                        break
                
                if found_column:
                    print(f"Using '{found_column}' instead of 'file_path'")
                    file_paths = []
                    csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
                    for row in csv_reader:
                        if row[found_column]:
                            file_paths.append(row[found_column].strip())
                else:
                    # If we can't find a column header, try using the first column
                    print("No 'file_path' column found. Trying first column...")
                    
                    # Reset and parse as simple CSV without headers
                    csv_reader = csv.reader(StringIO(text_content), delimiter=delimiter)
                    file_paths = []
                    for row in csv_reader:
                        if row and row[0].strip():
                            file_paths.append(row[0].strip())
            else:
                # Standard case - 'file_path' column exists
                file_paths = []
                for row in csv_reader:
                    if row['file_path'] and row['file_path'].strip():
                        file_paths.append(row['file_path'].strip())
        except Exception as parse_error:
            print(f"Error parsing CSV with delimiter '{delimiter}': {parse_error}")
            
            # Last resort - try splitting the content by each line and assuming one path per line
            file_paths = []
            for line in text_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.lower().startswith('file_path'):
                    file_paths.append(line)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="No valid file paths found in file")
        
        print(f"Found {len(file_paths)} image paths in file. First few: {file_paths[:5]}")
        
        # Load images from paths
        loaded_images = []
        for path in file_paths:
            try:
                # Check if path is absolute or relative
                if not os.path.isabs(path):
                    # Try relative to current directory
                    path = os.path.join(os.getcwd(), path)
                
                if not os.path.exists(path):
                    print(f"Warning: File not found: {path}")
                    continue
                
                # Load image
                img = Image.open(path).convert('RGB')
                img_tensor = al_manager.transform(img)
                
                # Add to unlabeled data
                img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)
                al_manager.unlabeled_data[img_id] = img_tensor
                loaded_images.append(img_id)
                
            except Exception as e:
                print(f"Error loading image from {path}: {str(e)}")
                continue
        
        if not loaded_images:
            raise HTTPException(status_code=400, detail="Failed to load any valid images from the provided paths")
        
        # Process the loaded images similarly to add_initial_data
        total_images = len(loaded_images)
        val_size = int(total_images * val_split)
        initial_labeled_size = int((total_images - val_size) * initial_labeled_ratio)
        
        # Shuffle and split
        np.random.shuffle(loaded_images)
        val_indices = loaded_images[:val_size]
        initial_labeled_indices = loaded_images[val_size:val_size + initial_labeled_size]
        unlabeled_indices = loaded_images[val_size + initial_labeled_size:]
        
        # Move images to appropriate sets
        for idx in val_indices:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.validation_data[idx] = (img_tensor, None)
            
        for idx in initial_labeled_indices:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.labeled_data[idx] = (img_tensor, None)  # These will need manual labeling
        
        split_info = {
            "total_images": total_images,
            "validation": len(val_indices),
            "initial_labeled": len(initial_labeled_indices),
            "unlabeled": len(unlabeled_indices)
        }
        
        return {
            "status": "success",
            "message": f"Successfully loaded {total_images} images from file",
            "split_info": split_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
def create_consistent_label_mapping(csv_content, label_column, delimiter, predefined_labels=None):
    """
    Create a consistent label mapping that respects predefined label order
    """
    import csv
    from io import StringIO
    
    # If we have predefined labels (from frontend), use their order
    if predefined_labels:
        label_to_index = {label: idx for idx, label in enumerate(predefined_labels)}
        return label_to_index
    
    # Otherwise, collect all unique labels from CSV and sort them
    csv_reader = csv.DictReader(StringIO(csv_content), delimiter=delimiter)
    unique_labels = set()
    
    for row in csv_reader:
        label_str = row.get(label_column, "").strip()
        if label_str:
            unique_labels.add(label_str)
    
    # Sort labels alphabetically for consistency (AMD comes before DR)
    sorted_labels = sorted(list(unique_labels))
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    
    return label_to_index


@app.post("/upload-combined-with-labels")
async def upload_combined_with_labels(
    files: List[UploadFile],
    val_split: float = Form(0.2),
    initial_labeled_ratio: float = Form(0.4),
    label_column: str = Form("label") # Default label column name
):
    """
    Process a combined upload of a CSV file with both file paths and class labels, plus image files
    """
    try:
        # Separate CSV and image files
        csv_files = [f for f in files if f.filename.endswith(('.csv', '.tsv', '.txt'))]
        image_files = [f for f in files if f.content_type and f.content_type.startswith('image/')]
        
        if not csv_files:
            raise HTTPException(status_code=400, detail="No CSV file found in upload")
        
        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in upload")
        
        # Read the CSV file
        csv_file = csv_files[0]
        csv_content = await csv_file.read()
        csv_text = csv_content.decode('utf-8', errors='replace')
        
        # Parse CSV to get file paths and labels
        import csv
        from io import StringIO
        
        # Try to identify the delimiter
        first_line = csv_text.split('\n')[0]
        delimiter = ',' # default
        for potential_delimiter in [',', '\t', ';', '|']:
            if potential_delimiter in first_line:
                delimiter = potential_delimiter
                break
        
        # Create a dictionary of image filename to file data
        image_map = {}
        for img_file in image_files:
            # Store by full name and by name without path
            image_map[img_file.filename] = img_file
            base_name = os.path.basename(img_file.filename)
            image_map[base_name] = img_file
        
        # First pass: identify the file path and label columns
        csv_reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
        file_path_column = None
        label_column_name = label_column  # Start with the provided label column
        
        # Common column names for file paths
        path_column_names = ['file_path', 'filepath', 'path', 'filename', 'image', 'file']
        # Common column names for labels
        label_column_names = ['label', 'class', 'category', 'target', 'y', 'classification']
        
        # Identify the file path column
        for field in csv_reader.fieldnames:
            if field.lower() in path_column_names:
                file_path_column = field
                break
        
        # If no label column name was provided or found, try to identify it
        if label_column_name not in csv_reader.fieldnames:
            for field in csv_reader.fieldnames:
                if field.lower() in label_column_names:
                    label_column_name = field
                    break
        
        if not file_path_column:
            raise HTTPException(status_code=400, detail="Could not find file path column in CSV")
            
        if label_column_name not in csv_reader.fieldnames:
            print(f"Warning: No label column found. Available columns: {csv_reader.fieldnames}")
            has_labels = False
        else:
            has_labels = True
            print(f"Found label column: {label_column_name}")
        
        # Restart the reader
        csv_reader = csv.DictReader(StringIO(csv_text), delimiter=delimiter)
        
        # Process each row and match with images
        labeled_images = []  # Images with labels from CSV
        unlabeled_images = []  # Images without labels
        label_to_index = {}  # Map string labels to numeric indices

        

        
        for row in csv_reader:
            path = row.get(file_path_column, "").strip()
            if not path:
                continue
                
            # Extract the filename from the path
            filename = os.path.basename(path)
            
            # Look up the image in our map
            if filename in image_map:
                img_file = image_map[filename]
                content = await img_file.read()  # Read the content of the file
                
                try:
                    # Convert to PIL Image and then to tensor
                    img = Image.open(io.BytesIO(content)).convert('RGB')
                    img_tensor = al_manager.transform(img)
                    
                    # Generate a unique image ID
                    img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)

                    # Check if this image has a label
                    if has_labels:
                        label_str = row.get(label_column_name, "").strip()
                        if label_str:
                            # Convert string label to numeric index if we haven't seen it before
                            if label_str not in label_to_index:
                                label_to_index[label_str] = len(label_to_index)
                            
                            label_idx = label_to_index[label_str]
                            # Add to labeled data
                            al_manager.labeled_data[img_id] = (img_tensor, label_idx)
                            labeled_images.append(img_id)
                            continue
                    
                    # If we get here, the image is unlabeled
                    al_manager.unlabeled_data[img_id] = img_tensor
                    unlabeled_images.append(img_id)
                    
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    continue
        
        if not (labeled_images or unlabeled_images):
            raise HTTPException(status_code=400, detail="Failed to process any images from the upload")
        
        print(f"Processed {len(labeled_images)} labeled images and {len(unlabeled_images)} unlabeled images")
        print(f"Label mapping: {label_to_index}")
        
        # Take some unlabeled images for validation
        val_size = int(len(unlabeled_images) * val_split)
        val_indices = unlabeled_images[:val_size]
        remaining_unlabeled = unlabeled_images[val_size:]
        
        # Move validation images to validation set
        for idx in val_indices:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.validation_data[idx] = (img_tensor, None)
        
        # Return the label mapping for future reference
        return {
            "status": "success",
            "message": f"Successfully processed images with labels from CSV",
            "stats": {
                "labeled": len(labeled_images),
                "unlabeled": len(remaining_unlabeled),
                "validation": len(val_indices),
                "total": len(labeled_images) + len(remaining_unlabeled) + len(val_indices)
            },
            "label_mapping": label_to_index
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV with labels: {str(e)}")

@app.post("/debug-csv-file")
async def debug_csv_file(csv_file: UploadFile = File(...)):
    """
    Debug CSV file to examine the paths and content
    """
    try:
        # Read CSV content
        content = await csv_file.read()
        text_content = content.decode('utf-8', errors='replace')
        
        # Parse CSV
        import csv
        from io import StringIO
        
        # Try comma as default delimiter
        delimiter = ','
        if '\t' in text_content[:1000]:  # Check first 1000 chars for tabs
            delimiter = '\t'
        
        csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
        
        # Get sample rows and column names
        sample_rows = []
        for i, row in enumerate(csv_reader):
            if i < 5:  # Get first 5 rows
                sample_rows.append(dict(row))
            else:
                break
                
        # Check if there's a file_path and annotation column
        columns = csv_reader.fieldnames if csv_reader.fieldnames else []
        
        # Add current directory info
        cwd = os.getcwd()
        search_dirs = [
            cwd,
            os.path.join(cwd, 'data'),
            os.path.join(cwd, 'images'),
            os.path.join(cwd, 'uploads')
        ]
        
        return {
            "columns": columns,
            "sample_rows": sample_rows,
            "delimiter_used": delimiter,
            "current_directory": cwd,
            "search_directories": search_dirs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV debug failed: {str(e)}")

@app.post("/upload-csv-paths-with-labels")
async def upload_csv_paths_with_labels(
    csv_file: UploadFile = File(...),
    label_column: str = Form(default="annotation"),
    delimiter: str = Form(default=","),
    val_split: float = Form(default=0.2),
    initial_labeled_ratio: float = Form(default=0.4),
    expected_label_mapping: str = Form(default=None)
):
    """
    Process a CSV file with both image paths and labels with consistent label mapping
    """
    try:
        # Validate inputs
        print(f"Received request:")
        print(f"  File: {csv_file.filename}")
        print(f"  Label column: {label_column}")
        print(f"  Delimiter: {delimiter}")
        print(f"  Expected label mapping: {expected_label_mapping}")
        
        # Parse expected label mapping if provided
        label_to_index = {}
        if expected_label_mapping and expected_label_mapping.strip():
            try:
                import json
                label_to_index = json.loads(expected_label_mapping)
                print(f"Using expected label mapping from frontend: {label_to_index}")
            except Exception as e:
                print(f"Error parsing expected label mapping: {e}")
                label_to_index = {}
        
        # Validate file
        if not csv_file or not csv_file.filename:
            raise HTTPException(status_code=400, detail="No CSV file provided")
            
        # Read CSV content
        try:
            content = await csv_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
                
            text_content = content.decode('utf-8', errors='replace')
            print(f"Successfully read {len(text_content)} characters from CSV")
            
        except Exception as read_error:
            print(f"Error reading CSV file: {str(read_error)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(read_error)}")
        
        # Handle special delimiter characters
        if delimiter == "\\t" or delimiter.lower() == "tab":
            delimiter = "\t"
        
        # Parse CSV
        import csv
        from io import StringIO
        
        try:
            csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
            fieldnames = csv_reader.fieldnames
            
            if not fieldnames:
                raise HTTPException(status_code=400, detail="CSV file has no headers")
                
            print(f"Found CSV columns: {fieldnames}")
                
        except Exception as csv_error:
            print(f"Error parsing CSV: {str(csv_error)}")
            raise HTTPException(status_code=400, detail=f"Error parsing CSV with delimiter '{delimiter}': {str(csv_error)}")
        
        # Find file path column
        file_path_column = None
        path_column_names = ['file_path', 'filepath', 'path', 'filename', 'image', 'file']
        
        for field in fieldnames:
            if field.lower() in path_column_names:
                file_path_column = field
                break
        
        if not file_path_column and len(fieldnames) > 0:
            file_path_column = fieldnames[0]
            print(f"Using first column as file path: '{file_path_column}'")
        
        if not file_path_column:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not identify file path column. Available columns: {fieldnames}"
            )
        
        # Find label column
        if label_column not in fieldnames:
            # Try common label column names
            label_alternatives = ['annotation', 'label', 'class', 'category', 'target', 'classification']
            found_alternative = None
            
            for field in fieldnames:
                if field.lower() in label_alternatives:
                    found_alternative = field
                    break
            
            if found_alternative:
                print(f"Using '{found_alternative}' instead of '{label_column}' for labels")
                label_column = found_alternative
            elif len(fieldnames) > 1:
                label_column = fieldnames[1]
                print(f"Using second column '{label_column}' as label column")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Label column '{label_column}' not found. Available columns: {fieldnames}"
                )
        
        print(f"Using file path column: '{file_path_column}'")
        print(f"Using label column: '{label_column}'")
        
        # If no expected mapping provided, create one by scanning all labels first
        if not label_to_index:
            print("No expected mapping provided. Scanning CSV for all labels...")
            csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
            unique_labels = set()
            
            for row in csv_reader:
                label_str = row.get(label_column, "").strip()
                if label_str:
                    unique_labels.add(label_str)
            
            # Create mapping with alphabetical order (AMD before DR)
            sorted_labels = sorted(list(unique_labels))
            label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
            print(f"Created alphabetical label mapping: {label_to_index}")
        
        # Process the CSV data
        csv_reader = csv.DictReader(StringIO(text_content), delimiter=delimiter)
        
        labeled_images = []
        unlabeled_images = []
        failed_paths = []
        
        # Define search directories
        import os
        cwd = os.getcwd()
        search_dirs = [
            cwd,
            os.path.join(cwd, 'data'),
            os.path.join(cwd, 'images'),
            os.path.join(cwd, 'uploads'),
            os.path.join(cwd, 'static'),
        ]
        
        print(f"Searching for images in: {search_dirs}")
        
        processed_count = 0
        for row_index, row in enumerate(csv_reader):
            if not row or all(not val for val in row.values()):
                continue
                
            file_path = row.get(file_path_column, "").strip()
            if not file_path:
                continue
                
            # Get label if it exists
            label_str = row.get(label_column, "").strip() if label_column in row else None
            
            # Try to find the image file
            found_path = None
            filename = os.path.basename(file_path)
            
            # Try different path combinations
            paths_to_try = [
                file_path,  # Original path
                *[os.path.join(d, filename) for d in search_dirs],  # Just filename in search dirs
                *[os.path.join(d, file_path) for d in search_dirs],  # Full path in search dirs
            ]
            
            for test_path in paths_to_try:
                if os.path.exists(test_path):
                    found_path = test_path
                    break
            
            if not found_path:
                failed_paths.append(file_path)
                continue
            
            # Load the image
            try:
                from PIL import Image
                img = Image.open(found_path).convert('RGB')
                img_tensor = al_manager.transform(img)
                
                # Generate unique image ID
                img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)
                
                # Process based on whether we have a label
                if label_str:
                    # Use the label mapping (either from frontend or created from CSV)
                    if label_str in label_to_index:
                        label_idx = label_to_index[label_str]
                        al_manager.labeled_data[img_id] = (img_tensor, label_idx)
                        labeled_images.append(img_id)
                        print(f"Labeled image {img_id} as '{label_str}' (index {label_idx})")
                    else:
                        print(f"Warning: Label '{label_str}' not found in mapping {label_to_index}")
                        # Add to unlabeled data instead
                        al_manager.unlabeled_data[img_id] = img_tensor
                        unlabeled_images.append(img_id)
                else:
                    al_manager.unlabeled_data[img_id] = img_tensor
                    unlabeled_images.append(img_id)
                
                processed_count += 1
                
            except Exception as img_error:
                print(f"Error processing image {found_path}: {str(img_error)}")
                failed_paths.append(file_path)
                continue
        
        if processed_count == 0:
            raise HTTPException(status_code=400, detail="Could not process any images")
        
        # Handle validation set
        if len(labeled_images) > 0:
            val_size = int(len(labeled_images) * val_split)
            if val_size > 0:
                val_indices = labeled_images[:val_size]
                remaining_labeled = labeled_images[val_size:]
                
                # Move validation images to validation set (keep their labels!)
                for idx in val_indices:
                    if idx in al_manager.labeled_data:
                        img_tensor, label = al_manager.labeled_data.pop(idx)
                        al_manager.validation_data[idx] = (img_tensor, label)
                
                labeled_images = remaining_labeled
        
        # Handle unlabeled images for validation if needed
        val_size_unlabeled = int(len(unlabeled_images) * val_split)
        val_indices_unlabeled = unlabeled_images[:val_size_unlabeled] if val_size_unlabeled > 0 else []
        remaining_unlabeled = unlabeled_images[val_size_unlabeled:] if val_size_unlabeled > 0 else unlabeled_images
        
        # Move unlabeled validation images
        for idx in val_indices_unlabeled:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.validation_data[idx] = (img_tensor, None)
        
        print(f"Final label mapping used: {label_to_index}")
        print(f"Processing complete: {len(labeled_images)} labeled, {len(remaining_unlabeled)} unlabeled, {len(al_manager.validation_data)} validation")
        
        return {
            "status": "success",
            "message": f"Successfully processed {processed_count} images from CSV",
            "stats": {
                "labeled": len(labeled_images),
                "unlabeled": len(remaining_unlabeled),
                "validation": len(al_manager.validation_data),
                "validation_labeled": len([1 for _, label in al_manager.validation_data.values() if label is not None]),
                "total": processed_count,
                "failed": len(failed_paths)
            },
            "label_mapping": label_to_index,
            "failed_paths": failed_paths[:10] if failed_paths else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload_csv_paths_with_labels: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recover-batch")
async def recover_batch(request: BatchRequest):
    """Emergency batch recovery with simple random sampling"""
    try:
        print("\n=== EMERGENCY BATCH RECOVERY ===")
        print(f"Using strategy: {request.strategy}, batch size: {request.batch_size}")
        
        # Make sure we have unlabeled data
        if len(al_manager.unlabeled_data) == 0:
            raise HTTPException(status_code=400, detail="No unlabeled data available")
            
        # Use a limited batch size for safety
        batch_size = min(request.batch_size, len(al_manager.unlabeled_data))
        
        # Simple random sampling directly
        image_ids = list(al_manager.unlabeled_data.keys())
        selected_ids = random.sample(image_ids, batch_size)
        
        selected_samples = []
        for img_id in selected_ids:
            selected_samples.append({
                "image_id": img_id,
                "uncertainty": 0.5,  # Default uncertainty
                "predictions": [{"label": "Unknown", "confidence": 0.0}]
            })
        
        al_manager.current_batch = [x["image_id"] for x in selected_samples]
        
        print(f"Successfully recovered batch with {len(selected_samples)} images")
        return selected_samples
        
    except Exception as e:
        print(f"Error in recover_batch: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def main():
    return {"message": "Welcome to Active Learning API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)