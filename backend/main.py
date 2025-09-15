from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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
from fastapi.responses import Response
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
import zipfile
import tempfile
import shutil
import pandas as pd
from datetime import datetime

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

class SimpleViTClassifier(nn.Module):
    """Simple ViT-based classifier for RETFound and similar models"""
    def __init__(self, num_classes=2, feature_dim=768):
        super().__init__()
        # This will be replaced with loaded ViT weights
        self.feature_extractor = nn.Identity()  # Placeholder
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        # This is a placeholder - will be replaced when loading RETFound weights
        features = self.feature_extractor(x)
        if len(features.shape) > 2:
            features = features.mean(dim=1)  # Global average pooling
        return self.classifier(features)

def create_vit_model(num_classes=2, image_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
    """Create a proper Vision Transformer model"""
    try:
        # Try to use timm if available for better ViT support
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        return model
    except ImportError:
        # Fallback to a custom ViT implementation
        return SimpleViTClassifier(num_classes=num_classes, feature_dim=embed_dim)

class ImprovedViTClassifier(nn.Module):
    """Improved ViT-based classifier"""
    def __init__(self, num_classes=2, image_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification head (use cls token)
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        
        return x

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
        self.image_paths = {}
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
                
            # Set up project
            self.project_name = project_name
            self.output_dir = os.path.join("output", project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(self.output_dir, exist_ok=True)
                
            # Initialize model based on type with enhanced custom support
            if model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, num_classes)
            elif model_name == "vision-transformer" or model_name == "vit":
                # Create a proper ViT architecture
                self.model = create_vit_model(num_classes=num_classes)
            elif model_name == "custom":
                # For custom models, start with a flexible base that can be adapted
                print(f"Initializing custom model architecture for {num_classes} classes")
                self.model = self._create_custom_model(num_classes)
            else:
                # Handle other model types or treat as custom
                print(f"Model type '{model_name}' not explicitly supported, treating as custom")
                self.model = self._create_custom_model(num_classes, model_type=model_name)
                
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

    def _create_custom_model(self, num_classes, model_type="custom"):
        """
        Create a flexible custom model that can be adapted to different architectures
        """
        try:
            # Try to create different types of custom models based on hints
            if "vit" in model_type.lower() or "transformer" in model_type.lower():
                return self._create_custom_vit(num_classes)
            elif "resnet" in model_type.lower():
                return self._create_custom_resnet(num_classes)
            elif "efficientnet" in model_type.lower():
                return self._create_custom_efficientnet(num_classes)
            else:
                # Default flexible architecture
                return self._create_flexible_custom_model(num_classes)
                
        except Exception as e:
            print(f"Error creating custom model: {e}")
            # Fallback to a simple ResNet50
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            return model

    def _create_custom_vit(self, num_classes):
        """Create a custom Vision Transformer model"""
        return ImprovedViTClassifier(
            num_classes=num_classes,
            image_size=224,
            patch_size=16,
            embed_dim=768,
            num_heads=12,
            num_layers=12
        )

    def _create_custom_resnet(self, num_classes):
        """Create a custom ResNet-style model"""
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    def _create_custom_efficientnet(self, num_classes):
        """Create a custom EfficientNet-style model"""
        try:
            # Try to use timm for EfficientNet
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
            return model
        except ImportError:
            print("timm not available, falling back to ResNet")
            return self._create_custom_resnet(num_classes)

    def _create_flexible_custom_model(self, num_classes):
        """
        Create a flexible custom model that can adapt to different state dicts
        """
        class FlexibleCustomModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Start with a basic CNN backbone
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                
                # Flexible classifier
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return FlexibleCustomModel(num_classes)

    def load_custom_model_weights(self, state_dict, num_classes=None):
        """
        Load weights into a custom model with flexible adaptation
        """
        try:
            # First, try to determine the model type from the state dict
            model_type = self._detect_model_type_from_state_dict(state_dict)
            
            if num_classes is None:
                num_classes = self._detect_num_classes_from_state_dict(state_dict)
            
            # Create appropriate model based on detected type
            if model_type == "vit":
                self.model = self._create_custom_vit(num_classes)
            elif model_type == "resnet":
                self.model = self._create_custom_resnet(num_classes)
            else:
                self.model = self._create_flexible_custom_model(num_classes)
            
            # Try to load the state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading custom model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading custom model: {unexpected_keys}")
                
            self.model = self.model.to(self.device)
            return True
            
        except Exception as e:
            print(f"Error loading custom model weights: {e}")
            return False

    def _detect_model_type_from_state_dict(self, state_dict):
        """Detect model type from state dict keys"""
        keys = list(state_dict.keys())
        
        # Check for ViT indicators
        if any(key in str(keys) for key in ['cls_token', 'pos_embed', 'patch_embed']):
            return "vit"
        
        # Check for ResNet indicators
        if any(key.startswith('layer') for key in keys):
            return "resnet"
        
        # Check for other architectures
        if 'features' in str(keys) and 'classifier' in str(keys):
            return "cnn"
        
        return "unknown"

    def _detect_num_classes_from_state_dict(self, state_dict):
        """Detect number of classes from the final layer"""
        # Look for common final layer patterns
        final_layer_patterns = ['fc.weight', 'classifier.weight', 'head.weight']
        
        for pattern in final_layer_patterns:
            if pattern in state_dict:
                return state_dict[pattern].shape[0]
        
        # Look for any layer ending with these patterns
        for key in state_dict.keys():
            if key.endswith('.weight') and any(pattern.split('.')[0] in key for pattern in final_layer_patterns):
                return state_dict[key].shape[0]
        
        return 2  # Default fallback

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

    def evaluate_model_on_unlabeled(self, num_samples=10):
        """
        Evaluate model performance on a sample of unlabeled data
        Returns predictions with confidence scores for the next batch of images
        """
        try:
            if not self.model or len(self.unlabeled_data) == 0:
                return None
                
            self.model.eval()
            
            # Get a sample of unlabeled data
            sample_size = min(num_samples, len(self.unlabeled_data))
            sample_ids = list(self.unlabeled_data.keys())[:sample_size]
            
            predictions = []
            all_confidences = []
            
            with torch.no_grad():
                for img_id in sample_ids:
                    img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    
                    # Get top prediction
                    top_prob, top_class = torch.max(probs, dim=1)
                    confidence = float(top_prob.item())
                    predicted_class = int(top_class.item())
                    
                    # Get all class probabilities
                    all_probs = []
                    for i, prob in enumerate(probs[0]):
                        all_probs.append({
                            'class_index': i,
                            'probability': float(prob.item())
                        })
                    
                    # Sort by probability (highest first)
                    all_probs.sort(key=lambda x: x['probability'], reverse=True)
                    
                    predictions.append({
                        'image_id': img_id,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'all_probabilities': all_probs
                    })
                    
                    all_confidences.append(confidence)
            
            # Calculate overall statistics
            overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return {
                'predictions': predictions,
                'overall_confidence': overall_confidence,
                'num_evaluated': len(predictions),
                'episode_info': {
                    'episode': self.episode,
                    'validation_accuracy': self.best_val_acc
                }
            }
            
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return None

    def get_evaluation_batch(self, num_samples=10):
        """
        Get the next batch of unlabeled images for evaluation display
        Similar to get_next_batch but focused on evaluation metrics
        """
        try:
            if not self.model or len(self.unlabeled_data) == 0:
                return None
                
            # Get evaluation data
            evaluation_data = self.evaluate_model_on_unlabeled(num_samples)
            
            if evaluation_data:
                # Add uncertainty scores for each prediction
                for pred in evaluation_data['predictions']:
                    # Calculate uncertainty (1 - confidence)
                    pred['uncertainty'] = 1 - pred['confidence']
                    
                    # Add prediction list in the format expected by the UI
                    pred['predictions'] = [
                        {
                            'label': f"Class {i}",  # This will be updated with actual labels by the frontend
                            'confidence': prob['probability']
                        }
                        for i, prob in enumerate(pred['all_probabilities'])
                    ]
            
            return evaluation_data
            
        except Exception as e:
            print(f"Error getting evaluation batch: {str(e)}")
            return None
        
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
            
            # STORE THE ORIGINAL FILENAME
            self.image_paths[img_id] = img_file.filename or f"uploaded_image_{img_id}.jpg"

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

    def set_custom_model(self, model, model_name="custom"):
        """
        Set a custom model directly (for when users provide their own model)
        """
        try:
            self.model = model.to(self.device)
            self.model_name = model_name
            
            # Reinitialize optimizer with new model
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001)
            )
            
            print(f"Custom model '{model_name}' set successfully")
            return True
            
        except Exception as e:
            print(f"Error setting custom model: {e}")
            return False

    def adapt_model_for_classes(self, num_classes):
        """
        Adapt the current model for a different number of classes
        """
        try:
            if hasattr(self.model, 'fc'):
                # ResNet-style models
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.model, 'classifier'):
                # ViT or other models with classifier
                if isinstance(self.model.classifier, nn.Linear):
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)
                elif isinstance(self.model.classifier, nn.Sequential):
                    # Find the last Linear layer
                    for i, layer in enumerate(reversed(self.model.classifier)):
                        if isinstance(layer, nn.Linear):
                            in_features = layer.in_features
                            self.model.classifier[-i-1] = nn.Linear(in_features, num_classes)
                            break
            elif hasattr(self.model, 'head'):
                # ViT head
                in_features = self.model.head.in_features
                self.model.head = nn.Linear(in_features, num_classes)
            else:
                print("Warning: Could not adapt model for new number of classes")
                return False
                
            self.model = self.model.to(self.device)
            
            # Reinitialize optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001)
            )
            
            return True
            
        except Exception as e:
            print(f"Error adapting model for {num_classes} classes: {e}")
            return False

    def get_model_info(self):
        """
        Get information about the current model
        """
        if not self.model:
            return None
            
        try:
            info = {
                'model_class': self.model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(next(self.model.parameters()).device),
                'model_type': 'custom'  # Default to custom
            }
            
            # Try to determine model type
            if hasattr(self.model, 'fc'):
                info['final_layer'] = 'fc'
                info['num_classes'] = self.model.fc.out_features
            elif hasattr(self.model, 'classifier'):
                info['final_layer'] = 'classifier'
                if isinstance(self.model.classifier, nn.Linear):
                    info['num_classes'] = self.model.classifier.out_features
            elif hasattr(self.model, 'head'):
                info['final_layer'] = 'head'
                info['num_classes'] = self.model.head.out_features
                
            return info
            
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None

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
        """Perform validation on the validation set - improved for CSV uploads"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        # Get validation data that has labels
        labeled_validation = {}
        for idx, (tensor, label) in self.validation_data.items():
            if label is not None:
                labeled_validation[idx] = (tensor, label)
        
        # If no labeled validation data, use a portion of labeled training data for validation
        if len(labeled_validation) == 0 and len(self.labeled_data) > 0:
            print("No labeled validation data found. Using portion of training data for validation.")
            # Use 20% of labeled data for validation
            val_size = max(1, len(self.labeled_data) // 5)
            val_items = list(self.labeled_data.items())[:val_size]
            labeled_validation = dict(val_items)
        
        if len(labeled_validation) == 0:
            print("Warning: No validation data available")
            return 0.0
        
        batch_size = 32
        with torch.no_grad():
            for idx, (img_tensor, label) in labeled_validation.items():
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += 1
                    
        validation_accuracy = 100.0 * total_correct / total_samples
        print(f"Validation Accuracy: {validation_accuracy:.2f}% ({total_correct}/{total_samples})")
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

            # Try to generate evaluation data
            evaluation_data = None
            try:
                print("Generating evaluation data for model assessment...")
                evaluation_data = self.get_evaluation_batch(num_samples=10)
                if evaluation_data:
                    print(f"Generated evaluation data for {len(evaluation_data['predictions'])} images")
                else:
                    print("No evaluation data could be generated")
            except Exception as e:
                print(f"Warning: Could not generate evaluation data: {str(e)}")

            # Select next batch if training successful and no evaluation data
            try:
                if evaluation_data is None:
                    # Get next batch using current model (original behavior)
                    next_batch = self.get_next_batch(
                        strategy=self.training_config["sampling_strategy"],
                        batch_size=batch_size
                    )
                else:
                    # If we have evaluation data, we'll let the frontend handle the next batch
                    next_batch = None
                    
                # Update episode metrics
                episode_metrics = {
                    'episode': self.episode,
                    'train_result': train_result,
                    'batch_size': len(next_batch) if next_batch else 0,
                    'strategy': self.training_config["sampling_strategy"],
                    'labeled_size': len(self.labeled_data),
                    'unlabeled_size': len(self.unlabeled_data),
                    'best_val_acc': best_val_acc,
                    'learning_rate': new_lr,
                    'lr_history': lr_history
                }
                
                self.episode_history.append(episode_metrics)
                
                # Update episode tracking
                self.plot_episode_xvalues.append(self.episode)
                self.plot_episode_yvalues.append(best_val_acc)
                
                # Save episode checkpoint
                if hasattr(self, 'checkpoint_manager') and self.checkpoint_manager:
                    try:
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
                    except Exception as checkpoint_error:
                        print(f"Warning: Failed to save episode checkpoint: {str(checkpoint_error)}")

                # Increment episode counter
                self.episode += 1
                
                # Return result with evaluation data if available
                result = {
                    "status": "success",
                    "metrics": episode_metrics,
                    "final_val_acc": best_val_acc  # Add this for backward compatibility
                }
                
                if evaluation_data:
                    result["evaluation_data"] = evaluation_data
                    print("Returning episode result with evaluation data")
                else:
                    result["next_batch"] = next_batch
                    print("Returning episode result with next batch")
                    
                return result
                    
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
            is_vit = any(k in ['cls_token', 'pos_embed', 'patch_embed'] for k in model_state.keys())
        
            if is_vit:
                print("Detected ViT model (like RETFound). Performing specialized adaptation...")
                
                # For ViT models, we want to:
                # 1. Load the feature extraction layers
                # 2. Replace the classification head with our own
                
                # Remove the original head if it exists
                keys_to_remove = [k for k in model_state.keys() if 'head' in k.lower()]
                for key in keys_to_remove:
                    print(f"Removing original head layer: {key}")
                    del model_state[key]
                
                # Load the feature extraction weights
                missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
                print(f"Loaded ViT weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                
                # Freeze feature extraction layers if requested
                if freeze_layers:
                    for name, param in self.model.named_parameters():
                        if 'classifier' not in name and 'head' not in name:
                            param.requires_grad = False
                            
                    print("Froze ViT feature extraction layers, keeping classifier trainable")
                
                return True
            
            # Continue with standard adaptation
            # Load the pretrained weights if not a transformer or transformer adaptation failed
            else:
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
        # ENSURE DIRECTORY EXISTS
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state, is_best=False):
        """Save checkpoint with correct episode numbering and better error handling"""
        try:
            # ENSURE DIRECTORY EXISTS BEFORE SAVING
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Use the episode from the state, not increment it
            episode = state.get('episode', 0)
            
            # For regular checkpoints, use the current episode
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode:03d}.pt')
            
            # If this is marked as best, also save as best
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pt')
                
            # Ensure the state includes all required fields with safe defaults
            safe_state = {
                'episode': episode,
                'model_state': state.get('model_state'),
                'best_val_acc': state.get('best_val_acc', 0),
                'training_config': state.get('training_config', {}),
                'labeled_indices': state.get('labeled_indices', []),
                'unlabeled_indices': state.get('unlabeled_indices', []),
                'validation_indices': state.get('validation_indices', []),
                'metrics': state.get('metrics', {}),
                'episode_history': state.get('episode_history', [])
            }
            
            # Add optimizer state if available
            if 'optimizer_state' in state and state['optimizer_state']:
                safe_state['optimizer_state'] = state['optimizer_state']
            
            # Always include scheduler_state to avoid KeyError later (even if empty)
            if 'scheduler_state' in state and state['scheduler_state']:
                safe_state['scheduler_state'] = state['scheduler_state']
            else:
                safe_state['scheduler_state'] = {}
            
            # Save the checkpoint with better error handling
            try:
                torch.save(safe_state, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                
                # Save as best if requested
                if is_best:
                    torch.save(safe_state, best_path)
                    print(f"Best model saved to: {best_path}")
                
                return checkpoint_path
                
            except Exception as save_error:
                print(f"Error saving checkpoint file: {str(save_error)}")
                # Try to save with a different filename if the original fails
                alternative_path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode}_{int(time.time())}.pt')
                try:
                    torch.save(safe_state, alternative_path)
                    print(f"Checkpoint saved to alternative path: {alternative_path}")
                    return alternative_path
                except Exception as alt_error:
                    print(f"Alternative save also failed: {str(alt_error)}")
                    raise save_error
            
        except Exception as e:
            print(f"Error in save_checkpoint: {str(e)}")
            # Don't let checkpoint errors break training
            print("Warning: Checkpoint save failed, but continuing training...")
            return None
            
    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load model checkpoint with safe handling of missing components"""
        try:
            if checkpoint_path is None:
                # Find the latest checkpoint
                checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_ep*.pt'))
                if not checkpoints:
                    print("No checkpoints found")
                    return None
                checkpoint_path = max(checkpoints, key=os.path.getctime)
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Load model state
            if 'model_state' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state'])
                    print("Model state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load model state: {e}")
                    # Try loading with strict=False
                    try:
                        missing, unexpected = model.load_state_dict(checkpoint['model_state'], strict=False)
                        print(f"Model loaded with missing keys: {missing}, unexpected keys: {unexpected}")
                    except Exception as e2:
                        print(f"Error: Could not load model state even with strict=False: {e2}")
                        return None
            else:
                print("Warning: No model state found in checkpoint")
            
            # Load optimizer state if available and provided
            if optimizer is not None and 'optimizer_state' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("Optimizer state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            elif optimizer is not None:
                print("No optimizer state found in checkpoint")
            
            # Load scheduler state SAFELY - only if both scheduler exists and checkpoint has scheduler_state
            if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state']:
                try:
                    # Check if the scheduler_state is not empty
                    scheduler_state = checkpoint['scheduler_state']
                    if scheduler_state and isinstance(scheduler_state, dict) and scheduler_state:
                        # Check if scheduler has the load_state_dict method
                        if hasattr(scheduler, 'load_state_dict'):
                            scheduler.load_state_dict(scheduler_state)
                            print("Scheduler state loaded successfully")
                        elif hasattr(scheduler, 'scheduler') and hasattr(scheduler.scheduler, 'load_state_dict'):
                            # Handle wrapped schedulers
                            if 'scheduler_state' in scheduler_state:
                                scheduler.scheduler.load_state_dict(scheduler_state['scheduler_state'])
                            else:
                                scheduler.scheduler.load_state_dict(scheduler_state)
                            print("Wrapped scheduler state loaded successfully")
                        else:
                            print("Warning: Scheduler does not support state loading")
                    else:
                        print("Scheduler state is empty, skipping")
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
            elif scheduler is not None:
                print("No scheduler state found in checkpoint or scheduler_state is empty")
            
            print(f"Checkpoint loaded from episode {checkpoint.get('episode', 'unknown')}")
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
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
        
        # Only trigger training if we have completed the FULL batch AND aren't already training
        batch_is_complete = self.current_batch_labeled_count >= self.current_batch_size
        has_enough_samples = len(self.al_manager.labeled_data) >= self.min_required_samples
        
        if (batch_is_complete and has_enough_samples and not self.is_training):
            print("Batch is complete! Starting training cycle...")
            asyncio.create_task(self._train_and_get_next_batch())
        else:
            if self.is_training:
                print("Training already in progress...")
            elif not batch_is_complete:
                print(f"Waiting for more labels before training ({self.current_batch_labeled_count}/{self.current_batch_size})")
            elif not has_enough_samples:
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
        """Training cycle with improved state management and evaluation"""
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
            
            # **CHECK FOR EVALUATION DATA**
            if 'evaluation_data' in training_result and training_result['evaluation_data']:
                print("Evaluation data found - evaluation screen should be shown")
                # Don't get next batch automatically - let the evaluation screen handle it
                self.is_training = False
                self.last_training_start = None
                return {
                    "status": "success",
                    "evaluation_available": True,
                    "evaluation_data": training_result['evaluation_data'],
                    "training_result": training_result,
                    "validation_accuracy": self.al_manager.best_val_acc
                }
            
            # If no evaluation data, continue with normal batch flow
            if not self.stop_requested:
                print("No evaluation data - getting next batch...")
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

def get_model_num_classes(model):
    """
    Safely get the number of output classes from different model architectures
    """
    try:
        # ResNet models
        if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            return model.fc.out_features
        
        # ViT models or custom models with classifier
        elif hasattr(model, 'classifier'):
            if hasattr(model.classifier, 'out_features'):
                return model.classifier.out_features
            elif isinstance(model.classifier, nn.Sequential):
                # Look for the last Linear layer in Sequential classifier
                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        
        # Look for head layer (common in ViT)
        elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            return model.head.out_features
        
        # Custom ViT classifier
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            return model.classifier.out_features
        
        # Sequential model - look for last Linear layer
        elif isinstance(model, nn.Sequential):
            for layer in reversed(model):
                if isinstance(layer, nn.Linear):
                    return layer.out_features
        
        # If all else fails, try to infer from the model's forward pass
        # This is more risky but can work as a last resort
        print(f"Warning: Could not determine num_classes for model type {type(model)}")
        print(f"Model structure: {model}")
        
        # Default fallback
        return 2
        
    except Exception as e:
        print(f"Error getting num_classes: {str(e)}")
        return 2

@app.get("/export-project")
async def export_project():
    """Export complete project as ZIP file with model, data, and metadata"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
        
        print("Starting project export...")
        
        # INSPECT THE MODEL AND SAVE INFO
        model_info = inspect_and_save_model_info(al_manager.model, "current_model_inspection.json")
        print("=== MODEL INSPECTION COMPLETE ===")
        print(f"Class name: {model_info.get('basic_info', {}).get('class_name', 'unknown')}")
        print(f"Detected type: {model_info.get('detection_result', {}).get('detected_type', 'unknown')}")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            print("Created ZIP buffer...")
            
            # Add the model inspection JSON to the ZIP
            with open("current_model_inspection.json", 'r') as f:
                zipf.writestr("model_inspection.json", f.read())
            
            # 1. Export model
            # Use the inspection results for better type detection
            model_type_name = model_info.get('detection_result', {}).get('detected_type', 'custom')
            if model_type_name == 'resnet':
                model_type_name = model_info.get('detection_result', {}).get('variant', 'resnet50')
            elif model_type_name == 'unknown':
                model_type_name = 'custom'
            
            num_classes = get_model_num_classes(al_manager.model)
            is_vit = model_type_name == 'vision_transformer'
            
            model_export = {
                'model_state': al_manager.model.state_dict(),
                'model_config': {
                    'project_name': al_manager.project_name,
                    'episode': al_manager.episode,
                    'model_type': model_type_name,
                    'model_class': model_info.get('basic_info', {}).get('class_name', 'unknown'),
                    'num_classes': num_classes,
                    'best_val_acc': al_manager.best_val_acc,
                    'is_vision_transformer': is_vit,
                    'detection_confidence': model_info.get('detection_result', {}).get('confidence', 'unknown'),
                    'detection_reasoning': model_info.get('detection_result', {}).get('reasoning', [])
                }
            }
        
            # Save model to temporary bytes buffer
            model_buffer = io.BytesIO()
            torch.save(model_export, model_buffer)
            model_buffer.seek(0)
            zipf.writestr("model.pt", model_buffer.getvalue())
            print("Added model to ZIP...")
            
            # 2. Create annotated CSV with ALL images and their REAL paths
            csv_data = []
            
            # Helper function to get real image path
            def get_image_info(img_id, img_tensor, label, split_type):
                # Get the actual stored path or create a fallback
                original_path = al_manager.image_paths.get(img_id, f"image_{img_id}.jpg")
                
                return {
                    'image_id': img_id,
                    'image_path': original_path,
                    'status': 'labeled' if label is not None else 'unlabeled',
                    'label_index': label,
                    'label_name': None,  # Will be filled below if needed
                    'split': split_type
                }
            
            # Add labeled images
            for img_id, (img_tensor, label) in al_manager.labeled_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, label, 'train'))
            
            # Add unlabeled images
            for img_id, img_tensor in al_manager.unlabeled_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, None, 'train'))
            
            # Add validation images
            for img_id, (img_tensor, label) in al_manager.validation_data.items():
                csv_data.append(get_image_info(img_id, img_tensor, label, 'validation'))
            
            # Fill in label names if we have current labels
            if hasattr(al_manager, 'current_labels') and al_manager.current_labels:
                for item in csv_data:
                    if item['label_index'] is not None and item['label_index'] < len(al_manager.current_labels):
                        item['label_name'] = al_manager.current_labels[item['label_index']]
            
            # Save CSV
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zipf.writestr("annotations.csv", csv_buffer.getvalue().encode('utf-8'))
                print("Added CSV with real paths to ZIP...")
            
            # 3. Create comprehensive metadata
            training_config = automated_trainer.training_config if 'automated_trainer' in globals() else {}
            
            # Get current labels from the frontend (stored by update-project-labels endpoint)
            current_labels = getattr(al_manager, 'current_labels', [])
            
            # If no current labels, create defaults based on detected num_classes
            if not current_labels:
                current_labels = [f"Class {i + 1}" for i in range(num_classes)]
            
            print(f"Exporting with labels: {current_labels}")
            
            metadata = {
               'project_info': {
                    'project_name': al_manager.project_name,
                    'export_timestamp': datetime.now().isoformat(),
                    # Export the last COMPLETED episode, not the next episode to run
                    'current_episode': max(0, al_manager.episode - 1),  # Subtract 1 because episode is "next to run"
                    'best_validation_accuracy': al_manager.best_val_acc,
                    'model_type': model_type_name,
                    'model_class': al_manager.model.__class__.__name__,
                    'is_vision_transformer': model_type_name == 'vision-transformer',
                    'num_classes': num_classes,
                },
                'labels': {
                    'label_names': current_labels,
                    'num_classes': len(current_labels)
                },
                'dataset_stats': {
                    'total_images': len(al_manager.labeled_data) + len(al_manager.unlabeled_data) + len(al_manager.validation_data),
                    'labeled_images': len(al_manager.labeled_data),
                    'unlabeled_images': len(al_manager.unlabeled_data),
                    'validation_images': len(al_manager.validation_data),
                    'validation_labeled': len([1 for _, label in al_manager.validation_data.values() if label is not None]),
                },
                'hyperparameters': {
                    'sampling_strategy': training_config.get('sampling_strategy', 'unknown'),
                    'batch_size': training_config.get('batch_size', 'unknown'),
                    'epochs': training_config.get('epochs', 'unknown'),
                    'learning_rate': training_config.get('learning_rate', 'unknown'),
                    'validation_split': al_manager.config.get('val_split', 0.2),
                    'initial_labeled_ratio': al_manager.config.get('initial_labeled_ratio', 0.1),
                },
                'training_metrics': {
                    'episode_accuracies': {
                        'episodes': al_manager.plot_episode_xvalues,
                        'accuracies': al_manager.plot_episode_yvalues
                    },
                    'epoch_losses': {
                        'epochs': al_manager.plot_epoch_xvalues,
                        'losses': al_manager.plot_epoch_yvalues
                    },
                    'episode_history': al_manager.episode_history
                },
                'episode_breakdown': []
            }
            
            # Add episode breakdown with detailed stats
            for i, episode_data in enumerate(al_manager.episode_history):
                episode_info = {
                    'episode': i + 1,
                    'strategy_used': episode_data.get('strategy', 'unknown'),
                    'batch_size': episode_data.get('batch_size', 'unknown'),
                    'images_labeled_this_episode': episode_data.get('batch_size', 0),
                    'total_labeled_after_episode': episode_data.get('labeled_size', 0),
                    'validation_accuracy': episode_data.get('best_val_acc', 0),
                    'learning_rate': episode_data.get('learning_rate', 'unknown')
                }
                metadata['episode_breakdown'].append(episode_info)
            
            # Save metadata to ZIP
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("metadata.json", metadata_json.encode('utf-8'))
            print(f"Added metadata to ZIP with labels: {current_labels}")
        
        # Prepare the ZIP file for download
        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()
        zip_filename = f"{al_manager.project_name}_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        print(f"ZIP file created successfully. Size: {len(zip_content)} bytes")
        
        # Return as streaming response
        return Response(
            content=zip_content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}",
                "Content-Length": str(len(zip_content))
            }
        )
        
    except Exception as e:
        print(f"Error exporting project: {str(e)}")
        import traceback
        traceback.print_exc()
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
    
def train_episode(self, epochs: int, batch_size: int, learning_rate: float):
    """Run a complete training episode with improved batch selection, checkpointing, LR scheduling, and evaluation"""
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
                
                # Try to save checkpoint but don't fail training if it doesn't work
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
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(state, is_best=True)
                        if checkpoint_path:
                            print(f"Checkpoint saved successfully: {checkpoint_path}")
                        else:
                            print("Warning: Checkpoint save failed, but training continues...")
                except Exception as checkpoint_error:
                    print(f"Warning: Failed to save checkpoint: {str(checkpoint_error)}")
                    print("Training will continue without checkpoint...")

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

        # **NEW: Get evaluation data on next unlabeled images**
        evaluation_data = None
        try:
            evaluation_data = self.get_evaluation_batch(num_samples=10)
            if evaluation_data:
                print(f"Generated evaluation data for {len(evaluation_data['predictions'])} images")
        except Exception as e:
            print(f"Warning: Could not generate evaluation data: {str(e)}")

        # Select next batch if training successful and no evaluation data
        try:
            if evaluation_data is None:
                # Get next batch using current model (original behavior)
                next_batch = self.get_next_batch(
                    strategy=self.training_config["sampling_strategy"],
                    batch_size=batch_size
                )
            else:
                # If we have evaluation data, we'll let the frontend handle the next batch
                next_batch = None
                
            # Update episode metrics
            episode_metrics = {
                'episode': self.episode,
                'train_result': train_result,
                'batch_size': len(next_batch) if next_batch else 0,
                'strategy': self.training_config["sampling_strategy"],
                'labeled_size': len(self.labeled_data),
                'unlabeled_size': len(self.unlabeled_data),
                'best_val_acc': best_val_acc,
                'learning_rate': new_lr,
                'lr_history': lr_history
            }
            
            self.episode_history.append(episode_metrics)
            
            # Update episode tracking
            self.plot_episode_xvalues.append(self.episode)
            self.plot_episode_yvalues.append(best_val_acc)
            
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
            
            # **NEW: Return evaluation data if available**
            result = {
                "status": "success",
                "metrics": episode_metrics,
                "final_val_acc": best_val_acc  # Add this for backward compatibility
            }
            
            if evaluation_data:
                result["evaluation_data"] = evaluation_data
                print("Returning episode result with evaluation data")
            else:
                result["next_batch"] = next_batch
                print("Returning episode result with next batch")
                
            return result
                
        except Exception as e:
            raise ValueError(f"Error after training: {str(e)}")
            
    except Exception as e:
        print(f"Error in train_episode: {str(e)}")
        raise

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
    """Save current model checkpoint manually"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        if not al_manager.output_dir:
            raise HTTPException(status_code=400, detail="No output directory set")
            
        # Initialize checkpoint manager if it doesn't exist
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        # Create the state to save with CURRENT episode (don't increment)
        state = {
            'episode': al_manager.episode,  # Current episode, not incremented
            'model_state': al_manager.model.state_dict(),
            'best_val_acc': al_manager.best_val_acc,
            'training_config': getattr(automated_trainer, 'training_config', {}),
            'labeled_indices': list(al_manager.labeled_data.keys()),
            'unlabeled_indices': list(al_manager.unlabeled_data.keys()),
            'validation_indices': list(al_manager.validation_data.keys()),
            'metrics': {
                'episode_accuracies': {
                    'x': al_manager.plot_episode_xvalues,
                    'y': al_manager.plot_episode_yvalues
                },
                'epoch_losses': {
                    'x': al_manager.plot_epoch_xvalues,
                    'y': al_manager.plot_epoch_yvalues
                }
            },
            'episode_history': al_manager.episode_history
        }
        
        # Add optimizer and scheduler states safely
        if hasattr(al_manager, 'optimizer') and al_manager.optimizer:
            try:
                state['optimizer_state'] = al_manager.optimizer.state_dict()
            except Exception as e:
                print(f"Warning: Could not save optimizer state: {e}")
                state['optimizer_state'] = {}
        else:
            state['optimizer_state'] = {}
            
        if hasattr(al_manager, 'lr_scheduler') and al_manager.lr_scheduler:
            try:
                if hasattr(al_manager.lr_scheduler, 'state_dict'):
                    state['scheduler_state'] = al_manager.lr_scheduler.state_dict()
                else:
                    state['scheduler_state'] = {}
            except Exception as e:
                print(f"Warning: Could not save scheduler state: {e}")
                state['scheduler_state'] = {}
        else:
            state['scheduler_state'] = {}
        
        checkpoint_path = al_manager.checkpoint_manager.save_checkpoint(state)
        
        return {
            "status": "success",
            "checkpoint_path": os.path.basename(checkpoint_path),
            "message": f"Checkpoint saved for episode {al_manager.episode}"
        }
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {str(e)}")

@app.post("/load-checkpoint")
async def load_checkpoint(request: Request):
    """Load model checkpoint with proper request handling"""
    try:
        # Parse the request body
        try:
            body = await request.json()
            checkpoint_path = body.get('checkpoint_path')
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON in request body: {str(e)}")
        
        if not checkpoint_path:
            raise HTTPException(status_code=422, detail="checkpoint_path is required")
        
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        if not al_manager.output_dir:
            raise HTTPException(status_code=400, detail="No output directory set")
            
        # Initialize checkpoint manager if it doesn't exist
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        # Construct full checkpoint path
        full_checkpoint_path = os.path.join(al_manager.checkpoint_manager.checkpoint_dir, checkpoint_path)
        
        if not os.path.exists(full_checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {full_checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = al_manager.checkpoint_manager.load_checkpoint(
            al_manager.model, 
            getattr(al_manager, 'optimizer', None),
            getattr(al_manager, 'lr_scheduler', None),
            full_checkpoint_path
        )
        
        if checkpoint:
            # Restore training state
            al_manager.episode = checkpoint.get('episode', 0)
            al_manager.best_val_acc = checkpoint.get('best_val_acc', 0)
            
            # Restore metrics
            metrics = checkpoint.get('metrics', {})
            al_manager.plot_episode_xvalues = metrics.get('episode_accuracies', {}).get('x', [])
            al_manager.plot_episode_yvalues = metrics.get('episode_accuracies', {}).get('y', [])
            al_manager.plot_epoch_xvalues = metrics.get('epoch_losses', {}).get('x', [])
            al_manager.plot_epoch_yvalues = metrics.get('epoch_losses', {}).get('y', [])
            
            # Restore episode history
            al_manager.episode_history = checkpoint.get('episode_history', [])
            
            # Restore data indices if available
            if 'labeled_indices' in checkpoint:
                # Note: We can't restore the actual data, but we can track the counts
                labeled_count = len(checkpoint['labeled_indices'])
                unlabeled_count = len(checkpoint['unlabeled_indices'])
                validation_count = len(checkpoint['validation_indices'])
                print(f"Checkpoint had {labeled_count} labeled, {unlabeled_count} unlabeled, {validation_count} validation images")
            
            return {
                "status": "success",
                "episode": checkpoint.get('episode', 0),
                "best_val_acc": checkpoint.get('best_val_acc', 0),
                "message": f"Checkpoint {checkpoint_path} loaded successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load checkpoint")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")
    
@app.get("/list-checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    try:
        if not al_manager.output_dir:
            return {"checkpoints": []}
            
        # Initialize checkpoint manager if it doesn't exist
        if not hasattr(al_manager, 'checkpoint_manager') or al_manager.checkpoint_manager is None:
            al_manager.checkpoint_manager = CheckpointManager(al_manager.output_dir)
            
        checkpoint_dir = al_manager.checkpoint_manager.checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            return {"checkpoints": []}
            
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*.pt'))
        checkpoint_names = [os.path.basename(cp) for cp in checkpoints]
        
        return {"checkpoints": sorted(checkpoint_names)}
        
    except Exception as e:
        print(f"Error listing checkpoints: {str(e)}")
        return {"checkpoints": []}

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
        
        # Validate model type - now including custom models
        supported_models = ["resnet18", "resnet50", "vision-transformer", "custom", "efficientnet", "densenet", "mobilenet"]
        
        if model_type not in supported_models:
            # If not in our supported list, automatically assign to "custom"
            print(f"Model type '{model_type}' not in supported list, treating as 'custom'")
            model_type = "custom"
        
        # Load model state
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Use safe model loading
        state_dict = safe_load_model(tmp_path)
        
        if state_dict is None:
            os.unlink(tmp_path)
            raise ValueError("Failed to load model file. The file may be corrupted or in an unsupported format.")
        
        # Extract the model state based on the structure
        model_state = extract_model_state(state_dict)
        
        # For custom models, try to auto-detect the architecture and classes
        if model_type == "custom":
            print("Processing custom model...")
            
            # Try to detect the model type and number of classes
            try:
                detected_info = analyze_model_structure(state_dict)
                print(f"Detected model info: {detected_info}")
                
                if detected_info["detected_type"] != "unknown":
                    print(f"Auto-detected model type: {detected_info['detected_type']}")
                    # You could optionally override the model_type here
                    # model_type = detected_info["detected_type"]
                
                if detected_info["num_classes"] and detected_info["num_classes"] != num_classes:
                    print(f"Auto-detected {detected_info['num_classes']} classes, updating from {num_classes}")
                    num_classes = detected_info["num_classes"]
                    
            except Exception as detection_error:
                print(f"Could not auto-detect model structure: {detection_error}")
        
        # Initialize project with the detected/specified model type
        if not al_manager.project_name:
            init_result = al_manager.initialize_project(
                project_name=project_name,
                model_name=model_type,  # This can now be "custom"
                num_classes=num_classes
            )
        
        # For custom models, use the enhanced loading method
        if model_type == "custom":
            success = al_manager.load_custom_model_weights(model_state, num_classes)
            if not success:
                # Fallback to standard loading
                try:
                    missing_keys, unexpected_keys = al_manager.model.load_state_dict(model_state, strict=False)
                    print(f"Loaded custom model with missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
                except Exception as load_error:
                    print(f"Custom model loading failed: {load_error}")
                    raise ValueError(f"Failed to load custom model: {load_error}")
        else:
            # Standard loading for known architectures
            try:
                al_manager.model.load_state_dict(model_state, strict=False)
                print("Loaded model state with non-strict matching")
            except Exception as e:
                print(f"Standard loading error: {str(e)}")
                
                # Try key remapping for common patterns
                fixed_state_dict = {}
                for k, v in model_state.items():
                    if k.startswith('module.'):
                        fixed_state_dict[k[7:]] = v
                    elif not k.startswith('module.') and f'module.{k}' in al_manager.model.state_dict():
                        fixed_state_dict[f'module.{k}'] = v
                    else:
                        fixed_state_dict[k] = v
                
                missing_keys, unexpected_keys = al_manager.model.load_state_dict(fixed_state_dict, strict=False)
                print(f"Loaded with key fixing. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # Clean up
        os.unlink(tmp_path)
        
        # Get model info for response
        model_info = al_manager.get_model_info()
        
        return {
            "status": "success",
            "message": f"{'Custom' if model_type == 'custom' else model_type.title()} model imported successfully",
            "project_name": project_name,
            "model_type": model_type,
            "num_classes": num_classes,
            "model_info": model_info,
            "detected_architecture": detected_info.get("detected_type", "unknown") if model_type == "custom" else model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
    
@app.post("/verify-custom-model")
async def verify_custom_model(uploaded_file: UploadFile = File(...)):
    """
    Verify and analyze a custom model file before importing
    """
    try:
        content = await uploaded_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Try to load and analyze the model
            state_dict = safe_load_model(tmp_path)
            
            if state_dict is None:
                os.unlink(tmp_path)
                return {
                    "status": "error",
                    "compatible": False,
                    "message": "Unable to load model file. File may be corrupted or in an unsupported format."
                }
            
            # Analyze the model structure
            model_info = analyze_model_structure(state_dict)
            
            os.unlink(tmp_path)
            
            return {
                "status": "success",
                "compatible": True,
                "analysis": model_info,
                "recommended_model_type": model_info.get("detected_type", "custom"),
                "detected_classes": model_info.get("num_classes"),
                "message": f"Custom model analysis complete. Detected: {model_info.get('detected_type', 'unknown')} architecture"
            }
            
        except Exception as analysis_error:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return {
                "status": "error", 
                "compatible": False,
                "message": f"Error analyzing custom model: {str(analysis_error)}"
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
        
        if not model_state or not isinstance(model_state, dict):
            result["message"] = "Unable to extract model state dictionary. Invalid format."
            return result
            
        # Print some keys for debugging
        print(f"Keys in model state: {list(model_state.keys())[:10]}...")
        
        # Check model type by examining keys
        key_set = set([k.split('.')[0] for k in model_state.keys()])
        
        # Detect Vision Transformer (ViT) models like RETFound
        if any(k in ['cls_token', 'pos_embed', 'patch_embed'] for k in key_set):
            result["detected_type"] = "vision-transformer"
            
            # For ViT models like RETFound, look for the head layer
            head_keys = [k for k in model_state.keys() if 'head' in k.lower() and 'weight' in k]
            if head_keys:
                head_key = head_keys[0]
                head_shape = model_state[head_key].shape
                print(f"Found ViT head layer: {head_key} with shape: {head_shape}")
                
                # For RETFound and similar models, the head might be for feature extraction
                # We should NOT use the original head size if it's 512 (feature dimension)
                if head_shape[0] == 512 or head_shape[0] > 100:
                    result["num_classes"] = None  # Don't auto-set large feature dimensions
                    result["message"] = f"Detected Vision Transformer (likely RETFound). Original head has {head_shape[0]} outputs (likely features). You can specify your desired number of classes."
                else:
                    result["num_classes"] = head_shape[0]
                    result["message"] = f"Detected Vision Transformer with {head_shape[0]} output classes."
            else:
                result["message"] = "Detected Vision Transformer. No classification head found - will add custom head."
                
        # Detect ResNet models
        elif any(k.startswith('layer') for k in model_state.keys()):
            result["detected_type"] = "resnet"
            # Look for fc layer
            if 'fc.weight' in model_state:
                fc_shape = model_state['fc.weight'].shape
                result["num_classes"] = fc_shape[0]
                result["message"] = f"Detected ResNet with {fc_shape[0]} output classes."
            else:
                result["message"] = "Detected ResNet architecture."
                
        # Other model types...
        elif 'features' in key_set and 'classifier' in key_set:
            result["detected_type"] = "vgg-style"
        elif 'blocks' in key_set:
            result["detected_type"] = "mobilenet"
        else:
            result["detected_type"] = "custom"
        
        # Model is compatible if it has weights
        result["compatible"] = any(k.endswith('.weight') for k in model_state.keys())
        result["adaptation_needed"] = True
        
        if not result["compatible"]:
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
                img = Image.open(found_path).convert('RGB')
                img_tensor = al_manager.transform(img)
                
                # Generate unique image ID
                img_id = len(al_manager.unlabeled_data) + len(al_manager.labeled_data) + len(al_manager.validation_data)
                
                # *** ADD THIS LINE: Store the original path ***
                al_manager.image_paths[img_id] = found_path
                
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
                
                # Move validation images to validation set (KEEP their labels for initial training!)
                for idx in val_indices:
                    if idx in al_manager.labeled_data:
                        img_tensor, label = al_manager.labeled_data.pop(idx)
                        al_manager.validation_data[idx] = (img_tensor, label)  # Keep the label!
                
                labeled_images = remaining_labeled
                print(f"Moved {len(val_indices)} labeled images to validation set WITH labels")
        
        # Handle unlabeled images for validation if needed
        val_size_unlabeled = int(len(unlabeled_images) * val_split)
        val_indices_unlabeled = unlabeled_images[:val_size_unlabeled] if val_size_unlabeled > 0 else []
        remaining_unlabeled = unlabeled_images[val_size_unlabeled:] if val_size_unlabeled > 0 else unlabeled_images
        
        # Move unlabeled validation images (these have no labels)
        for idx in val_indices_unlabeled:
            if idx in al_manager.unlabeled_data:
                img_tensor = al_manager.unlabeled_data.pop(idx)
                al_manager.validation_data[idx] = (img_tensor, None)  # No label
        
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
    
@app.post("/import-project")
async def import_project(uploaded_file: UploadFile = File(...)):
    """Import a complete project from ZIP file and prepare for continuation"""
    try:
        # Read ZIP file
        content = await uploaded_file.read()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save ZIP file temporarily
            zip_path = os.path.join(temp_dir, "imported_project.zip")
            with open(zip_path, "wb") as f:
                f.write(content)
            
            # Extract ZIP file
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            # Find the project files
            project_files = {}
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file in ['model.pt', 'annotations.csv', 'metadata.json']:
                        project_files[file] = os.path.join(root, file)
            
            # Validate required files
            required_files = ['model.pt', 'metadata.json']
            missing_files = [f for f in required_files if f not in project_files]
            if missing_files:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required files: {missing_files}"
                )
            
            # Load metadata
            with open(project_files['metadata.json'], 'r') as f:
                metadata = json.load(f)
            
            project_info = metadata['project_info']
            dataset_stats = metadata['dataset_stats']
            hyperparameters = metadata['hyperparameters']
            training_metrics = metadata['training_metrics']
            labels_info = metadata.get('labels', {})
            
            # Get model type information with better handling
            model_type = project_info.get('model_type', 'resnet50')
            is_vit = project_info.get('is_vision_transformer', False)
            model_class = project_info.get('model_class', '')
            
            print(f"Importing project with model_type: {model_type}, is_vit: {is_vit}, class: {model_class}")
            
            # Load model data
            if torch.cuda.is_available():
                model_data = torch.load(project_files['model.pt'])
            else:
                model_data = torch.load(project_files['model.pt'], map_location=torch.device('cpu'))
            
            model_config = model_data['model_config']
            model_state = model_data['model_state']
            
            # Additional check for ViT from model state if metadata doesn't have it
            if not is_vit and model_type not in ['vision-transformer', 'custom']:
                # Check model state for ViT indicators
                vit_indicators = ['cls_token', 'pos_embed', 'patch_embed']
                if any(key in model_state.keys() for key in vit_indicators):
                    print("Detected ViT model from state dict")
                    is_vit = True
                    model_type = 'vision-transformer'
            
            # Determine number of classes from model structure
            num_classes = determine_num_classes_from_state(model_state)
            if not num_classes:
                num_classes = project_info.get('num_classes', 2)
            
            print(f"Detected {num_classes} classes from model structure")
            
            # Initialize project with imported settings
            al_manager.project_name = project_info['project_name']
            imported_episode = project_info['current_episode']
            al_manager.episode = imported_episode + 1

            print(f"Imported project completed episode {imported_episode}, ready for episode {al_manager.episode}")

            al_manager.best_val_acc = project_info['best_validation_accuracy']
            
            # Update config
            al_manager.config.update({
                'val_split': hyperparameters.get('validation_split', 0.2),
                'initial_labeled_ratio': hyperparameters.get('initial_labeled_ratio', 0.1),
            })
            
            # Initialize the correct model type based on what was saved
            if is_vit or model_type == 'vision-transformer':
                print("Initializing Vision Transformer model")
                init_result = al_manager.initialize_project(
                    project_name=al_manager.project_name,
                    model_name='vision-transformer',
                    num_classes=num_classes
                )
            else:
                print(f"Initializing {model_type} model")
                
                # Map model type names for ResNet variants
                if 'resnet' in model_type.lower():
                    if '18' in model_type:
                        model_name = 'resnet18'
                    else:
                        model_name = 'resnet50'
                else:
                    model_name = 'resnet50'  # Default fallback
                
                init_result = al_manager.initialize_project(
                    project_name=al_manager.project_name,
                    model_name=model_name,
                    num_classes=num_classes
                )
            
            # Load model weights with structure adaptation
            try:
                al_manager.model.load_state_dict(model_state, strict=True)
                print("Model loaded with strict=True")
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Attempting to adapt model structure...")
                
                # Try to adapt the model state dict
                adapted_state = adapt_model_state_dict(model_state, al_manager.model.state_dict())
                missing_keys, unexpected_keys = al_manager.model.load_state_dict(adapted_state, strict=False)
                
                if missing_keys:
                    print(f"Missing keys after adaptation: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys after adaptation: {unexpected_keys}")
                
                print("Model loaded with adapted state dict")
            
            # Restore training metrics
            al_manager.plot_episode_xvalues = training_metrics['episode_accuracies']['episodes']
            al_manager.plot_episode_yvalues = training_metrics['episode_accuracies']['accuracies']
            al_manager.plot_epoch_xvalues = training_metrics['epoch_losses']['epochs']
            al_manager.plot_epoch_yvalues = training_metrics['epoch_losses']['losses']
            al_manager.episode_history = training_metrics.get('episode_history', [])
            
            # Update automated trainer config if it exists
            if 'automated_trainer' in globals():
                automated_trainer.training_config.update({
                    'sampling_strategy': hyperparameters.get('sampling_strategy', 'least_confidence'),
                    'epochs': hyperparameters.get('epochs', 10),
                    'batch_size': hyperparameters.get('batch_size', 32),
                    'learning_rate': hyperparameters.get('learning_rate', 0.001)
                })
            
            # Clear existing data sets - we're starting fresh with new images
            al_manager.labeled_data.clear()
            al_manager.unlabeled_data.clear()
            al_manager.validation_data.clear()
            
            # Initialize image_paths if it doesn't exist
            if not hasattr(al_manager, 'image_paths'):
                al_manager.image_paths = {}
            else:
                al_manager.image_paths.clear()
            
            # Create output directory
            al_manager.output_dir = os.path.join("output", al_manager.project_name, 
                datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(al_manager.output_dir, exist_ok=True)
            
            # **NEW: Try to load existing annotations automatically**
            loaded_images_count = 0
            project_ready = False
            
            if 'annotations.csv' in project_files:
                print("Found annotations.csv, attempting to load existing data...")
                try:
                    # Read the CSV file
                    import pandas as pd
                    df = pd.read_csv(project_files['annotations.csv'])
                    
                    print(f"Annotations CSV contains {len(df)} entries")
                    
                    # Try different common directory locations for images
                    search_paths = [
                        os.getcwd(),  # Current working directory
                        os.path.join(os.getcwd(), 'data'),
                        os.path.join(os.getcwd(), 'images'),
                        os.path.join(os.getcwd(), 'uploads'),
                        extract_dir,  # Inside the extracted project
                        os.path.dirname(project_files['annotations.csv'])  # Same dir as CSV
                    ]
                    
                    failed_paths = []
                    
                    for _, row in df.iterrows():
                        try:
                            original_path = row['image_path']
                            image_id = int(row['image_id'])
                            label_index = row['label_index'] if pd.notna(row['label_index']) else None
                            split_type = row['split']
                            
                            # Try to find the image in various locations
                            image_found = False
                            actual_path = None
                            
                            # First try the original path
                            if os.path.exists(original_path):
                                actual_path = original_path
                                image_found = True
                            else:
                                # Try just the filename in various search paths
                                filename = os.path.basename(original_path)
                                for search_path in search_paths:
                                    candidate_path = os.path.join(search_path, filename)
                                    if os.path.exists(candidate_path):
                                        actual_path = candidate_path
                                        image_found = True
                                        break
                            
                            if image_found:
                                # Load the image
                                img = Image.open(actual_path).convert('RGB')
                                img_tensor = al_manager.transform(img)
                                
                                # Store the actual path we found
                                al_manager.image_paths[image_id] = actual_path
                                
                                # Place in appropriate dataset
                                if split_type == 'validation':
                                    al_manager.validation_data[image_id] = (img_tensor, int(label_index) if label_index is not None else None)
                                elif label_index is not None:
                                    al_manager.labeled_data[image_id] = (img_tensor, int(label_index))
                                else:
                                    al_manager.unlabeled_data[image_id] = img_tensor
                                
                                loaded_images_count += 1
                            else:
                                failed_paths.append(original_path)
                                
                        except Exception as e:
                            print(f"Error loading image {row.get('image_path', 'unknown')}: {str(e)}")
                            failed_paths.append(row.get('image_path', 'unknown'))
                            continue
                    
                    print(f"Successfully loaded {loaded_images_count} images from annotations")
                    if failed_paths:
                        print(f"Failed to load {len(failed_paths)} images. First few: {failed_paths[:5]}")
                    
                    # Determine if project is ready
                    project_ready = loaded_images_count > 0
                    
                except Exception as csv_error:
                    print(f"Error loading annotations.csv: {str(csv_error)}")
                    import traceback
                    traceback.print_exc()
            
            # Store the correct model type for return
            final_model_type = 'vision-transformer' if is_vit else model_type
            
            # Prepare the final message
            if project_ready:
                message = f"Project '{al_manager.project_name}' ({final_model_type}) imported successfully with {loaded_images_count} existing images loaded. Project is ready for active learning!"
            else:
                message = f"Project '{al_manager.project_name}' ({final_model_type}) imported successfully. Upload new images to continue training with the existing model."
            
            return {
                "status": "success",
                "project_info": {
                    **project_info,
                    "model_type": final_model_type,  # Return the correct model type
                    "is_vision_transformer": is_vit,
                    "num_classes": num_classes
                },
                "dataset_stats": {
                    **dataset_stats,
                    "current_labeled": len(al_manager.labeled_data),
                    "current_unlabeled": len(al_manager.unlabeled_data),
                    "current_validation": len(al_manager.validation_data),
                    "loaded_from_annotations": loaded_images_count
                },
                "hyperparameters": hyperparameters,
                "labels": labels_info,
                "training_config": automated_trainer.training_config if 'automated_trainer' in globals() else {},
                "model_ready": True,
                "project_ready": project_ready,  # New flag
                "images_loaded": loaded_images_count > 0,
                "message": message
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error importing project: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to import project: {str(e)}")

def determine_num_classes_from_state(state_dict):
    """Determine number of classes from model state dict"""
    try:
        # Check for different classifier layer structures
        
        # ResNet style fc layers
        if 'fc.weight' in state_dict:
            return state_dict['fc.weight'].shape[0]
        elif 'fc.1.weight' in state_dict:
            return state_dict['fc.1.weight'].shape[0]
        elif 'fc.2.weight' in state_dict:
            return state_dict['fc.2.weight'].shape[0]
        
        # ViT style classifiers
        elif 'classifier.weight' in state_dict:
            return state_dict['classifier.weight'].shape[0]
        elif 'head.weight' in state_dict:
            return state_dict['head.weight'].shape[0]
        
        # Look for any layer with 'fc' in the name
        fc_keys = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
        if fc_keys:
            return state_dict[fc_keys[0]].shape[0]
        
        # Look for any layer with 'classifier' in the name
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if classifier_keys:
            return state_dict[classifier_keys[0]].shape[0]
        
        return None
    except Exception as e:
        print(f"Error determining num_classes: {e}")
        return None


def adapt_model_state_dict(saved_state, target_state):
    """Adapt saved model state dict to match target model structure"""
    adapted_state = {}
    
    print("Adapting model state dict...")
    print(f"Saved state keys sample: {list(saved_state.keys())[:10]}...")
    print(f"Target state keys sample: {list(target_state.keys())[:10]}...")
    
    # Copy all non-classifier layers directly
    for key in saved_state.keys():
        if not any(classifier_key in key for classifier_key in ['fc.', 'classifier.', 'head.']):
            if key in target_state:
                adapted_state[key] = saved_state[key]
            else:
                print(f"Skipping key not in target: {key}")
    
    # Handle classifier layer mapping
    target_classifier_keys = [k for k in target_state.keys() if any(c in k for c in ['fc.', 'classifier.', 'head.'])]
    saved_classifier_keys = [k for k in saved_state.keys() if any(c in k for c in ['fc.', 'classifier.', 'head.'])]
    
    print(f"Target classifier keys: {target_classifier_keys}")
    print(f"Saved classifier keys: {saved_classifier_keys}")
    
    # Map different classifier structures
    if 'fc.weight' in target_state and 'fc.bias' in target_state:
        # Target expects simple Linear layer
        if 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.1.weight']
            adapted_state['fc.bias'] = saved_state['fc.1.bias']
            print("Mapped fc.1 -> fc")
        elif 'fc.2.weight' in saved_state and 'fc.2.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.2.weight']
            adapted_state['fc.bias'] = saved_state['fc.2.bias']
            print("Mapped fc.2 -> fc")
        elif 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['fc.weight']
            adapted_state['fc.bias'] = saved_state['fc.bias']
            print("Direct fc mapping")
        elif 'classifier.weight' in saved_state and 'classifier.bias' in saved_state:
            adapted_state['fc.weight'] = saved_state['classifier.weight']
            adapted_state['fc.bias'] = saved_state['classifier.bias']
            print("Mapped classifier -> fc")
    
    elif 'fc.1.weight' in target_state and 'fc.1.bias' in target_state:
        # Target expects Sequential with Linear at index 1
        if 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['fc.1.weight'] = saved_state['fc.weight']
            adapted_state['fc.1.bias'] = saved_state['fc.bias']
            print("Mapped fc -> fc.1")
        elif 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['fc.1.weight'] = saved_state['fc.1.weight']
            adapted_state['fc.1.bias'] = saved_state['fc.1.bias']
            print("Direct fc.1 mapping")
    
    elif 'classifier.weight' in target_state and 'classifier.bias' in target_state:
        # Target expects ViT-style classifier
        if 'fc.weight' in saved_state and 'fc.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['fc.weight']
            adapted_state['classifier.bias'] = saved_state['fc.bias']
            print("Mapped fc -> classifier")
        elif 'fc.1.weight' in saved_state and 'fc.1.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['fc.1.weight']
            adapted_state['classifier.bias'] = saved_state['fc.1.bias']
            print("Mapped fc.1 -> classifier")
        elif 'classifier.weight' in saved_state and 'classifier.bias' in saved_state:
            adapted_state['classifier.weight'] = saved_state['classifier.weight']
            adapted_state['classifier.bias'] = saved_state['classifier.bias']
            print("Direct classifier mapping")
    
    print(f"Adapted state final keys: {len(adapted_state)} keys")
    return adapted_state

def evaluate_model_on_unlabeled(self, num_samples=10):
    """
    Evaluate model performance on a sample of unlabeled data
    Returns predictions with confidence scores for the next batch of images
    """
    try:
        if not self.model or len(self.unlabeled_data) == 0:
            return None
            
        self.model.eval()
        
        # Get a sample of unlabeled data
        sample_size = min(num_samples, len(self.unlabeled_data))
        sample_ids = list(self.unlabeled_data.keys())[:sample_size]
        
        predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for img_id in sample_ids:
                img_tensor = self.unlabeled_data[img_id].unsqueeze(0).to(self.device)
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get top prediction
                top_prob, top_class = torch.max(probs, dim=1)
                confidence = float(top_prob.item())
                predicted_class = int(top_class.item())
                
                # Get all class probabilities
                all_probs = []
                for i, prob in enumerate(probs[0]):
                    all_probs.append({
                        'class_index': i,
                        'probability': float(prob.item())
                    })
                
                # Sort by probability (highest first)
                all_probs.sort(key=lambda x: x['probability'], reverse=True)
                
                predictions.append({
                    'image_id': img_id,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': all_probs
                })
                
                all_confidences.append(confidence)
        
        # Calculate overall statistics
        overall_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            'predictions': predictions,
            'overall_confidence': overall_confidence,
            'num_evaluated': len(predictions),
            'episode_info': {
                'episode': self.episode,
                'validation_accuracy': self.best_val_acc
            }
        }
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None
    
def get_evaluation_batch(self, num_samples=10):
    """
    Get the next batch of unlabeled images for evaluation display
    Similar to get_next_batch but focused on evaluation metrics
    """
    try:
        if not self.model or len(self.unlabeled_data) == 0:
            return None
            
        # Get evaluation data
        evaluation_data = self.evaluate_model_on_unlabeled(num_samples)
        
        if evaluation_data:
            # Add uncertainty scores for each prediction
            for pred in evaluation_data['predictions']:
                # Calculate uncertainty (1 - confidence)
                pred['uncertainty'] = 1 - pred['confidence']
                
                # Add prediction list in the format expected by the UI
                pred['predictions'] = [
                    {
                        'label': f"Class {i}",  # This will be updated with actual labels by the frontend
                        'confidence': prob['probability']
                    }
                    for i, prob in enumerate(pred['all_probabilities'])
                ]
        
        return evaluation_data
        
    except Exception as e:
        print(f"Error getting evaluation batch: {str(e)}")
        return None

def inspect_and_save_model_info(model, output_path="model_inspection.json"):
    """
    Comprehensive model inspection - saves all model info to JSON for debugging
    """
    try:
        state_dict = model.state_dict()
        
        # Collect all model information
        model_info = {
            "basic_info": {
                "class_name": model.__class__.__name__,
                "module_name": model.__class__.__module__,
                "model_type": str(type(model)),
                "model_repr": str(model)[:1000] + "..." if len(str(model)) > 1000 else str(model)
            },
            "state_dict_analysis": {
                "total_parameters": len(state_dict),
                "parameter_shapes": {k: list(v.shape) for k, v in state_dict.items()},
                "parameter_names": list(state_dict.keys()),
                "first_20_keys": list(state_dict.keys())[:20],
                "last_20_keys": list(state_dict.keys())[-20:]
            },
            "architecture_detection": {
                "has_resnet_layers": any(key.startswith('layer') for key in state_dict.keys()),
                "has_vit_indicators": any(indicator in key for key in state_dict.keys() 
                                        for indicator in ['cls_token', 'pos_embed', 'patch_embed', 'blocks']),
                "has_attention": any('attn' in key or 'attention' in key for key in state_dict.keys()),
                "has_transformer": any('transformer' in key for key in state_dict.keys()),
                "resnet_layer_keys": [k for k in state_dict.keys() if k.startswith('layer')],
                "vit_keys": [k for k in state_dict.keys() if any(indicator in k for indicator in ['cls_token', 'pos_embed', 'patch_embed', 'blocks'])],
                "attention_keys": [k for k in state_dict.keys() if 'attn' in k or 'attention' in k],
                "classifier_keys": [k for k in state_dict.keys() if any(c in k for c in ['classifier', 'head', 'fc'])]
            },
            "layer_analysis": {
                "conv_layers": [k for k in state_dict.keys() if 'conv' in k and 'weight' in k],
                "linear_layers": [k for k in state_dict.keys() if any(layer_type in k for layer_type in ['fc', 'linear', 'classifier', 'head']) and 'weight' in k],
                "norm_layers": [k for k in state_dict.keys() if any(norm_type in k for norm_type in ['bn', 'norm', 'layer_norm']) and 'weight' in k],
                "embedding_layers": [k for k in state_dict.keys() if 'embed' in k]
            },
            "model_structure": {},
            "pytorch_model_info": {}
        }
        
        # Try to get model structure
        try:
            # Get model children and modules
            children = list(model.children())
            named_children = list(model.named_children())
            modules = list(model.modules())
            named_modules = list(model.named_modules())
            
            model_info["model_structure"] = {
                "num_children": len(children),
                "num_modules": len(modules),
                "named_children": [(name, str(type(child))) for name, child in named_children],
                "child_types": [str(type(child)) for child in children],
                "has_fc": hasattr(model, 'fc'),
                "has_classifier": hasattr(model, 'classifier'),
                "has_head": hasattr(model, 'head'),
                "has_features": hasattr(model, 'features')
            }
            
            # Get specific layer info if they exist
            if hasattr(model, 'fc'):
                fc_layer = model.fc
                model_info["model_structure"]["fc_info"] = {
                    "type": str(type(fc_layer)),
                    "repr": str(fc_layer),
                    "has_in_features": hasattr(fc_layer, 'in_features'),
                    "has_out_features": hasattr(fc_layer, 'out_features'),
                    "in_features": getattr(fc_layer, 'in_features', None),
                    "out_features": getattr(fc_layer, 'out_features', None)
                }
            
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                model_info["model_structure"]["classifier_info"] = {
                    "type": str(type(classifier)),
                    "repr": str(classifier),
                    "has_in_features": hasattr(classifier, 'in_features'),
                    "has_out_features": hasattr(classifier, 'out_features'),
                    "in_features": getattr(classifier, 'in_features', None),
                    "out_features": getattr(classifier, 'out_features', None)
                }
            
            if hasattr(model, 'head'):
                head = model.head
                model_info["model_structure"]["head_info"] = {
                    "type": str(type(head)),
                    "repr": str(head),
                    "has_in_features": hasattr(head, 'in_features'),
                    "has_out_features": hasattr(head, 'out_features'),
                    "in_features": getattr(head, 'in_features', None),
                    "out_features": getattr(head, 'out_features', None)
                }
                
        except Exception as struct_error:
            model_info["model_structure"]["error"] = str(struct_error)
        
        # Try to get PyTorch model info
        try:
            import torch
            model_info["pytorch_model_info"] = {
                "device": str(next(model.parameters()).device) if list(model.parameters()) else "no_parameters",
                "dtype": str(next(model.parameters()).dtype) if list(model.parameters()) else "no_parameters",
                "requires_grad": any(p.requires_grad for p in model.parameters()),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "is_training": model.training
            }
        except Exception as torch_error:
            model_info["pytorch_model_info"]["error"] = str(torch_error)
        
        # Try to determine the actual model type
        detection_result = {
            "detected_type": "unknown",
            "confidence": "low",
            "reasoning": []
        }
        
        # ResNet detection
        if model_info["architecture_detection"]["has_resnet_layers"]:
            detection_result["detected_type"] = "resnet"
            detection_result["confidence"] = "high"
            detection_result["reasoning"].append("Found ResNet layer structure (layer1, layer2, etc.)")
            
            # Determine ResNet variant
            if model_info["model_structure"].get("fc_info", {}).get("in_features") == 512:
                detection_result["variant"] = "resnet18"
            elif model_info["model_structure"].get("fc_info", {}).get("in_features") == 2048:
                detection_result["variant"] = "resnet50"
        
        # ViT detection
        elif model_info["architecture_detection"]["has_vit_indicators"]:
            detection_result["detected_type"] = "vision_transformer"
            detection_result["confidence"] = "high"
            detection_result["reasoning"].append("Found ViT indicators (cls_token, pos_embed, patch_embed, blocks)")
        
        # Transformer detection
        elif model_info["architecture_detection"]["has_attention"]:
            detection_result["detected_type"] = "transformer"
            detection_result["confidence"] = "medium"
            detection_result["reasoning"].append("Found attention mechanisms")
        
        # Class name detection
        class_name = model_info["basic_info"]["class_name"].lower()
        if 'resnet' in class_name:
            detection_result["class_name_suggests"] = "resnet"
        elif any(vit_term in class_name for vit_term in ['vit', 'vision', 'transformer']):
            detection_result["class_name_suggests"] = "vision_transformer"
        else:
            detection_result["class_name_suggests"] = "unknown"
        
        model_info["detection_result"] = detection_result
        
        # Save to JSON file
        import json
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"Model inspection saved to: {output_path}")
        print(f"Detected model type: {detection_result['detected_type']} (confidence: {detection_result['confidence']})")
        
        return model_info
        
    except Exception as e:
        error_info = {
            "error": str(e),
            "basic_class_name": model.__class__.__name__ if hasattr(model, '__class__') else "unknown"
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"Error during model inspection: {e}")
        return error_info

def determine_model_type_for_export(model):
    """Determine the model type for export based on model structure"""
    
    # Get model state dict and class info
    state_dict = model.state_dict()
    class_name = model.__class__.__name__
    
    
    print(f"=== MODEL TYPE DETECTION DEBUG ===")
    print(f"Model class name: {class_name}")
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
    
    # Check model class name first (most reliable)
    if any(vit_indicator in class_name for vit_indicator in ['ViT', 'Vision', 'Transformer', 'SimpleViTClassifier']):
        print("Detected ViT from class name")
        return 'vision-transformer'
    
    # Check for ViT indicators in state dict keys
    vit_indicators = ['cls_token', 'pos_embed', 'patch_embed', 'blocks.', 'norm.weight', 'head.weight']
    found_vit_keys = [key for key in state_dict.keys() if any(indicator in key for indicator in vit_indicators)]
    
    if found_vit_keys:
        print(f"Detected ViT from state dict keys: {found_vit_keys[:5]}")
        return 'vision-transformer'
    
    # Check for transformer-like structure (attention layers)
    attention_keys = [key for key in state_dict.keys() if any(pattern in key for pattern in ['attn', 'attention', 'self_attention'])]
    if attention_keys:
        print(f"Detected transformer from attention keys: {attention_keys[:3]}")
        return 'vision-transformer'
    
    # Check for ResNet structure
    if any(key.startswith('layer') for key in state_dict.keys()):
        print("Detected ResNet from layer structure")
        # Determine ResNet variant
        feature_count = None
        for key in state_dict.keys():
            if ('fc' in key and 'weight' in key) or ('classifier' in key and 'weight' in key):
                weight_shape = state_dict[key].shape
                if len(weight_shape) == 2:
                    feature_count = weight_shape[1]
                    break
        
        if feature_count == 512:
            return 'resnet18'
        elif feature_count == 2048:
            return 'resnet50'
        else:
            return 'resnet50'  # Default
    
    # Check for other architectures
    if 'features' in [key.split('.')[0] for key in state_dict.keys()]:
        print("Detected VGG-style model")
        return 'vgg'
    
    # If we can't determine, check the final classifier structure
    classifier_keys = [k for k in state_dict.keys() if any(c in k for c in ['classifier', 'head', 'fc'])]
    if classifier_keys:
        print(f"Found classifier keys: {classifier_keys}")
        # If it has a simple classifier but no clear architecture, assume custom ViT
        if not any(key.startswith('layer') for key in state_dict.keys()):
            print("Assuming custom ViT due to classifier without ResNet layers")
            return 'vision-transformer'
    
    print("Defaulting to custom")
    return 'custom'

@app.post("/load-existing-annotations")
async def load_existing_annotations():
    """Load existing annotations from imported project and start active learning"""
    try:
        if not al_manager.project_name:
            raise HTTPException(status_code=400, detail="No project loaded")
        
        # Check if we have any data to work with
        total_data = len(al_manager.labeled_data) + len(al_manager.unlabeled_data) + len(al_manager.validation_data)
        
        if total_data == 0:
            return {
                "status": "no_data",
                "message": "No existing data found in project. Please upload new images."
            }
        
        # If we have unlabeled data, get a batch for active learning
        if len(al_manager.unlabeled_data) > 0:
            try:
                # Use the configured sampling strategy and batch size
                strategy = getattr(automated_trainer, 'training_config', {}).get('sampling_strategy', 'least_confidence')
                batch_size = getattr(automated_trainer, 'training_config', {}).get('batch_size', 32)
                
                batch = al_manager.get_next_batch(strategy, min(batch_size, len(al_manager.unlabeled_data)))
                
                return {
                    "status": "success", 
                    "message": f"Loaded existing project data. Ready for active learning with {len(batch)} images.",
                    "data_stats": {
                        "labeled": len(al_manager.labeled_data),
                        "unlabeled": len(al_manager.unlabeled_data), 
                        "validation": len(al_manager.validation_data)
                    },
                    "batch_ready": True
                }
            except Exception as e:
                return {
                    "status": "success",
                    "message": f"Loaded existing project data, but couldn't get batch: {str(e)}",
                    "data_stats": {
                        "labeled": len(al_manager.labeled_data),
                        "unlabeled": len(al_manager.unlabeled_data),
                        "validation": len(al_manager.validation_data)
                    },
                    "batch_ready": False
                }
        else:
            return {
                "status": "success",
                "message": "Project loaded successfully. All data is labeled - ready for final training.",
                "data_stats": {
                    "labeled": len(al_manager.labeled_data),
                    "unlabeled": len(al_manager.unlabeled_data),
                    "validation": len(al_manager.validation_data)
                },
                "batch_ready": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load existing annotations: {str(e)}")

@app.post("/update-project-labels")
async def update_project_labels(request: dict):
    """Update the current project labels"""
    try:
        labels = request.get('labels', [])
        # Store labels in the al_manager for export
        al_manager.current_labels = labels
        return {"status": "success", "labels": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate-model")
async def evaluate_model(num_samples: int = 10):
    """Get model evaluation on unlabeled data"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        evaluation_data = al_manager.evaluate_model_on_unlabeled(num_samples)
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="Unable to evaluate model - no unlabeled data available")
            
        return evaluation_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/evaluation-batch") 
async def get_evaluation_batch(num_samples: int = 10):
    """Get evaluation batch for display"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        evaluation_data = al_manager.get_evaluation_batch(num_samples)
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="No evaluation data available")
            
        return evaluation_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation batch: {str(e)}")

@app.post("/continue-from-evaluation")
async def continue_from_evaluation():
    """Continue training after evaluation screen"""
    try:
        if not al_manager.model:
            raise HTTPException(status_code=400, detail="No model initialized")
            
        # Get the next batch for continuing active learning
        if len(al_manager.unlabeled_data) == 0:
            return {
                "status": "complete",
                "message": "No more unlabeled data available. Training complete!"
            }
            
        # Get default training config
        strategy = getattr(automated_trainer, 'training_config', {}).get('sampling_strategy', 'least_confidence')
        batch_size = getattr(automated_trainer, 'training_config', {}).get('batch_size', 32)
        
        # Get next batch using active learning strategy
        batch = al_manager.get_next_batch(strategy, min(batch_size, len(al_manager.unlabeled_data)))
        
        return {
            "status": "success", 
            "message": f"Ready to continue with {len(batch)} new images for labeling",
            "batch_size": len(batch)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to continue from evaluation: {str(e)}")

@app.post("/train-episode")
async def train_episode(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train a single episode and get next batch"""
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
        
        # Train the episode using the existing train_episode method
        result = al_manager.train_episode(epochs, batch_size, learning_rate)
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Train episode endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def main():
    return {"message": "Welcome to Active Learning API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)