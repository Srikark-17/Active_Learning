from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from fastapi import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
import json
import time
from typing import List, Tuple
import types
import asyncio
import glob
import shutil

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
            # Validation checks
            if not project_name:
                raise ValueError("Project name is required")
            if model_name not in ["resnet18", "resnet50"]:
                raise ValueError("Model type must be either 'resnet18' or 'resnet50'")
            if num_classes <= 0:
                raise ValueError("Number of classes must be greater than 0")
                
            # Set up project
            self.project_name = project_name
            self.output_dir = os.path.join("output", project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.output_dir, exist_ok=True)
                
            # Initialize model first
            if model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif model_name == "resnet50":
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

        # Start with all data in unlabeled set
        for idx in unlabeled_indices:
            self.unlabeled_data[idx] = all_data[idx]

        # Move initial batch to labeled set
        for idx in initial_labeled_indices:
            self.validation_data[idx] = (all_data[idx], None)

        # Move validation data to validation set (they'll be labeled through the UI)
        for idx in val_indices:
            self.validation_data[idx] = (all_data[idx], None)

        # Save split information
        split_info = {
            "total_images": total_images,
            "validation": len(self.validation_data),
            "initial_labeled": initial_labeled_size,
            "unlabeled": len(self.unlabeled_data)
        }

        # Save splits to disk
        if self.output_dir:
            split_path = os.path.join(self.output_dir, 'data_splits.json')
            with open(split_path, 'w') as f:
                json.dump({
                    'validation_indices': val_indices,
                    'labeled_indices': initial_labeled_indices,
                    'unlabeled_indices': unlabeled_indices,
                    'split_info': split_info
                }, f)

        return split_info

    def get_next_batch(self, strategy: str, batch_size: int) -> List[dict]:
        """
        Select next batch of samples using specified strategy
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

        # Get uncertainty scores for all unlabeled samples
        sample_scores = self._get_sample_scores(strategy)
        
        # Select samples based on strategy
        selected_samples = self._select_samples(sample_scores, batch_size, strategy)
        
        # Update current batch
        self.current_batch = [x["image_id"] for x in selected_samples]
        
        return selected_samples
    
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
        """Validate model performance similar to train_al.py"""
        try:
            if len(self.validation_data) == 0:
                print("Warning: No validation data available")
                return 0.0
                    
            labeled_validation = [
                (img, label) for img, label in self.validation_data.values() 
                if label is not None
            ]
            
            if len(labeled_validation) == 0:
                print(f"Warning: No labeled validation data (0/{len(self.validation_data)} samples labeled)")
                return 0.0

            self.model.eval()
            total_correct = 0
            total_samples = 0

            batch_size = 32  # Process validation in batches like train_al.py
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
            raise

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
        self.training_config = {
            'sampling_strategy': None,
            'batch_size': None,
            'epochs': None,
            'learning_rate': 0.001
        }
        self.min_required_samples = 10
    
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
    """Get image data for display with improved error handling"""
    try:
        print(f"Serving image {image_id}")
        if image_id in al_manager.unlabeled_data:
            tensor = al_manager.unlabeled_data[image_id]
        elif image_id in al_manager.labeled_data:
            tensor = al_manager.labeled_data[image_id][0]
        elif image_id in al_manager.validation_data:
            tensor = al_manager.validation_data[image_id][0]
        else:
            print(f"Image {image_id} not found in any dataset")
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Ensure tensor is on CPU and in the right format
        tensor = tensor.cpu()
        
        # Denormalize the tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
            
        # Convert tensor to image
        img_array = tensor.numpy().transpose(1, 2, 0)
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        try:
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
            print(f"Error converting tensor to image: {str(e)}")
            raise HTTPException(status_code=500, detail="Error converting image")
            
    except Exception as e:
        print(f"Error serving image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Manually get next batch during automated training"""
    try:
        return await automated_trainer.get_new_batch()
    except Exception as e:
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

@app.get("/")
async def main():
    return {"message": "Welcome to Active Learning API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)