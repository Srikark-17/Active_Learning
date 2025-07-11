"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { RadioGroup, RadioGroupItem } from "../components/ui/radio-group";
import { AlertCircle, Upload, Download, Plus, Trash2 } from "lucide-react";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import ImageLoader from "../components/imageLoader";
import { Alert, AlertDescription } from "../components/ui/alert";
import activeLearnAPI from "../services/activelearning";
import EnhancedMetricsDisplay from "@/components/enhancedMetricsDisplay";
import AutomatedTrainingControls from "@/components/automatedTrainingControls";
import {
  ValidationProgress,
  BatchProgress,
  ModelPredictions,
  ActiveLearningStatus,
} from "@/components/components";
import CheckpointControls from "@/components/checkpointControls";
import PretrainedModelImport from "@/components/pretrainedModel";
import ModelAdaptationControls from "@/components/modelAdaptationControls";

const ActiveLearningUI = () => {
  // Project Configuration State
  const [projectName, setProjectName] = useState("");
  const [selectedModel, setSelectedModel] = useState("resnet50");
  const [currentImage, setCurrentImage] = useState("3559b.png");
  const [activeTab, setActiveTab] = useState("new");
  const [loadedImages, setLoadedImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [imageLoadError, setImageLoadError] = useState(null);

  // Active Learning State
  const [samplingStrategy, setSamplingStrategy] = useState("least_confidence");
  const [batchSize, setBatchSize] = useState(32);
  const [currentBatch, setCurrentBatch] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [isRetraining, setIsRetraining] = useState(false);
  const [labels, setLabels] = useState([]);
  const [isBatchInProgress, setIsBatchInProgress] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [episodeHistory, setEpisodeHistory] = useState([]);
  const [valSplit, setValSplit] = useState(0.2);
  const [initialLabeledRatio, setInitialLabeledRatio] = useState(0.4);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [automatedStatus, setAutomatedStatus] = useState(null);
  const [epochs, setEpochs] = useState(10);
  const [validationAccuracy, setValidationAccuracy] = useState(0);
  const [status, setStatus] = useState({
    project_name: null,
    current_episode: 0,
    labeled_count: 0,
    unlabeled_count: 0,
    validation_count: 0,
    current_batch_size: 0,
  });
  const [batchStats, setBatchStats] = useState({
    totalImages: batchSize,
    completed: 0,
    remaining: batchSize,
    accuracy: 0.85,
    timeElapsed: "00:00",
  });
  const [metrics, setMetrics] = useState({
    episodeAccuracies: [],
    currentEpochLosses: [],
  });
  const [validationStatus, setValidationStatus] = useState({
    total: 0,
    labeled: 0,
    unlabeled: 0,
    percent_labeled: 0,
  });
  const [lrConfig, setLrConfig] = useState({
    strategy: "plateau", // default strategy
    initial_lr: 0.001,
    factor: 0.1,
    patience: 5,
    min_lr: 1e-6,
  });
  const [lrHistory, setLrHistory] = useState();
  const fileDataRef = useRef(null);

  useEffect(() => {
    let interval;
    if (isRetraining) {
      interval = setInterval(async () => {
        try {
          const newMetrics = await activeLearnAPI.getMetrics();
          setMetrics({
            episodeAccuracies: newMetrics.episode_accuracies.x.map((x, i) => ({
              x,
              y: newMetrics.episode_accuracies.y[i],
            })),
            currentEpochLosses: newMetrics.current_epoch_losses.x.map(
              (x, i) => ({
                x,
                y: newMetrics.current_epoch_losses.y[i],
              })
            ),
          });
        } catch (error) {
          console.error("Failed to fetch metrics:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRetraining]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const [currentStatus, valStatus] = await Promise.all([
            activeLearnAPI.getStatus(),
            activeLearnAPI.getValidationStatus(), // Add this endpoint to your API service
          ]);
          setStatus(currentStatus);
          setValidationStatus(valStatus);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const currentStatus = await activeLearnAPI.getStatus();
          setStatus(currentStatus);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(
        async () => {
          try {
            const [status, metrics, history, lrStatus] = await Promise.all([
              activeLearnAPI.getAutomatedTrainingStatus(),
              activeLearnAPI.getMetrics(),
              activeLearnAPI.getEpisodeHistory(),
              activeLearnAPI.getLRSchedulerStatus(),
            ]);

            setAutomatedStatus(status);
            setTrainingMetrics(metrics);
            setEpisodeHistory(history.episodes);

            // Only update LR history if it exists
            if (lrStatus && lrStatus.history) {
              setLrHistory(lrStatus.history);
            }
          } catch (error) {
            console.error("Failed to fetch status:", error);
            // Don't break other functionality if LR status fails
            setLrHistory([]);
          }
        },
        status?.is_training ? 1000 : 5000
      );
    }
    return () => clearInterval(interval);
  }, [isInitialized, status?.is_training]);

  useEffect(() => {
    let interval;
    if (isRetraining) {
      interval = setInterval(async () => {
        try {
          const newMetrics = await activeLearnAPI.getMetrics();
          setMetrics(newMetrics); // Just pass the metrics directly without transformation
        } catch (error) {
          console.error("Failed to fetch metrics:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRetraining]);

  useEffect(() => {
    if (isInitialized && currentBatch.length > 0 && !isRetraining) {
      // Only get new batch if we haven't started labeling (completed === 0)
      if (batchStats.completed === 0) {
        getNextBatch();
        setBatchStats({
          totalImages: batchSize,
          completed: 0,
          remaining: batchSize,
          accuracy:
            metrics.episodeAccuracies[metrics.episodeAccuracies.length - 1]
              ?.y || 0.85,
          timeElapsed: "00:00",
        });
      }
    }
  }, [samplingStrategy, batchSize]);

  const handleAddLabel = () => {
    setLabels([...labels, ""]);
  };

  const handleLabelChange = (index, value) => {
    const newLabels = [...labels];
    newLabels[index] = value;
    setLabels(newLabels);
  };

  const handleRemoveLabel = (index) => {
    const newLabels = labels.filter((_, i) => i !== index);
    setLabels(newLabels);
  };

  const getNextBatch = async () => {
    try {
      console.log(
        `Getting next batch with strategy: ${samplingStrategy}, size: ${batchSize}`
      );

      // Validate batch size to avoid None or invalid values
      const validBatchSize = parseInt(batchSize) || 32;

      const batch = await activeLearnAPI.getBatch(
        samplingStrategy,
        validBatchSize
      );

      console.log(`Received batch with ${batch.length} images`);
      setCurrentBatch(batch);

      if (batch.length > 0) {
        // Preload the first image to check for errors
        const img = new Image();
        const imageUrl = activeLearnAPI.getImageUrl(batch[0].image_id);
        console.log(`Loading first image from URL: ${imageUrl}`);

        img.onload = () => {
          console.log("First image loaded successfully");
          setCurrentImage(imageUrl);
          setImageLoadError(null);
        };

        img.onerror = (e) => {
          console.error("Image loading error:", e);
          setImageLoadError(
            `Failed to load image (ID: ${batch[0].image_id}). Please check image paths.`
          );
        };

        img.src = imageUrl;
        setCurrentImageIndex(0);
      } else {
        setImageLoadError("Received empty batch. No images to display.");
      }
    } catch (error) {
      console.error("getBatch error:", error);
      setImageLoadError("Failed to get next batch: " + error.message);

      // Try fallback to random sampling
      try {
        console.log("Attempting fallback to random sampling");
        const fallbackBatch = await activeLearnAPI.getBatch(
          "random",
          Math.min(parseInt(batchSize) || 32, status.unlabeled_count || 10)
        );

        console.log(
          `Received fallback batch with ${fallbackBatch.length} images`
        );
        setCurrentBatch(fallbackBatch);

        if (fallbackBatch.length > 0) {
          const imageUrl = activeLearnAPI.getImageUrl(
            fallbackBatch[0].image_id
          );
          setCurrentImage(imageUrl);
          setCurrentImageIndex(0);
          setImageLoadError("Using random sampling as fallback");
        } else {
          setImageLoadError(
            "Failed to get any images. Try using a smaller batch size."
          );
        }
      } catch (fallbackError) {
        console.error("Fallback getBatch error:", fallbackError);
        setImageLoadError(
          "Failed to get batch even with fallback strategy. Try restarting the application."
        );
      }
    }
  };

  const handleImagesLoaded = async (
    files,
    uploadType = false,
    extraParam = ",",
    detectedLabels = []
  ) => {
    try {
      console.log("handleImagesLoaded called with:", {
        filesCount: files.length,
        uploadType,
        extraParam,
        detectedLabels,
        firstFile: files[0], // Log the actual file object
      });

      // Auto-populate labels if we have detected labels
      if (detectedLabels && detectedLabels.length > 0) {
        console.log("Auto-populating labels from CSV:", detectedLabels);
        const newLabels = detectedLabels
          .map((label) => label.trim())
          .filter((label) => label);

        if (newLabels.length > 0) {
          setLabels(newLabels);
        }
      }

      // Handle CSV-only upload with labels
      if (uploadType === "csv-with-labels") {
        const labelColumn = extraParam;
        const csvFile = files[0];

        if (!csvFile || !(csvFile instanceof File)) {
          throw new Error("Invalid CSV file object received");
        }

        // Store the file reference directly in ref
        fileDataRef.current = {
          type: "csv-with-labels",
          csvFile: csvFile,
          labelColumn: labelColumn,
          delimiter: ",",
          detectedLabels: detectedLabels,
        };

        // Store indicator in state (NOT the actual file)
        setLoadedImages({
          type: "csv-with-labels",
          fileCount: 1,
          labelColumn: labelColumn,
          delimiter: ",",
          detectedLabels: detectedLabels,
        });

        setImageLoadError(
          "CSV with labels processed. Click 'Start Project' to begin."
        );
        return;
      }

      // Store the loaded files first, before any API calls
      setLoadedImages(files);
      setImageLoadError(null);

      if (detectedLabels && detectedLabels.length > 0) {
        console.log("Auto-populating labels from CSV:", detectedLabels);
        const newLabels = detectedLabels
          .map((label) => label.trim())
          .filter((label) => label);

        if (newLabels.length > 0) {
          setLabels(newLabels);
        }
      }

      // Handle other upload types...
      if (uploadType === "combined-with-labels") {
        // Store files in ref for combined uploads too
        fileDataRef.current = {
          type: "combined-with-labels",
          files: files,
          labelColumn: extraParam,
          detectedLabels: detectedLabels,
        };

        setLoadedImages({
          type: "combined-with-labels",
          fileCount: files.length,
          labelColumn: extraParam,
          detectedLabels: detectedLabels,
        });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setImageLoadError("Failed to upload: " + error.message);
    }
  };

  const handleNextImage = () => {
    if (currentImageIndex < currentBatch.length - 1) {
      setCurrentImageIndex((prev) => prev + 1);
      setCurrentImage(
        activeLearnAPI.getImageUrl(currentBatch[currentImageIndex + 1].image_id)
      );
    }
  };

  const handlePreviousImage = () => {
    if (currentImageIndex > 0) {
      setCurrentImageIndex((prev) => prev - 1);
      setCurrentImage(
        activeLearnAPI.getImageUrl(currentBatch[currentImageIndex - 1].image_id)
      );
    }
  };
  const handleSubmitLabel = async () => {
    try {
      const imageId = currentBatch[currentImageIndex].image_id;
      const newCompleted = batchStats.completed + 1;
      const batch_complete = newCompleted === batchSize;

      const result = await activeLearnAPI.submitLabel(
        imageId,
        parseInt(selectedLabel)
      );

      setBatchStats((prev) => ({
        ...prev,
        completed: newCompleted,
        remaining: prev.totalImages - newCompleted,
      }));

      setSelectedLabel("");
      console.log(`Batch progress: ${newCompleted}/${batchSize}`);
      console.log("Batch complete?", batch_complete);

      if (batch_complete) {
        console.log("Starting episode training...");
        setIsRetraining(true);
        setImageLoadError("Batch complete - Starting episode training...");

        try {
          const episodeResult = await activeLearnAPI.startEpisode(
            epochs,
            batchSize
          );
          console.log("Episode training result:", episodeResult);

          if (episodeResult.final_val_acc) {
            setValidationAccuracy(episodeResult.final_val_acc);
          }

          await handleStartNewBatch();
          setImageLoadError("Episode training complete - New batch loaded");
          setIsRetraining(false);
        } catch (error) {
          console.error("Episode training error:", error);
          setImageLoadError("Episode training error: " + error.message);
          setIsRetraining(false);
        }
      } else {
        handleNextImage();
      }
    } catch (error) {
      setImageLoadError("Failed to submit label: " + error.message);
    }
  };

  const handleStartNewBatch = async () => {
    try {
      setImageLoadError("Getting next batch...");
      const batchResult = await getNextBatch();

      setBatchStats({
        totalImages: batchSize,
        completed: 0,
        remaining: batchSize,
        accuracy: validationAccuracy || batchResult.accuracy || 0,
        timeElapsed: "00:00",
      });

      setIsRetraining(false);
      setSelectedLabel("");
      setIsBatchInProgress(false);
      setCurrentImageIndex(0);
      setImageLoadError(null);
    } catch (error) {
      setImageLoadError("Failed to start new batch: " + error.message);
    }
  };

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(
        async () => {
          try {
            const [status, metrics, history] = await Promise.all([
              activeLearnAPI.getAutomatedTrainingStatus(),
              activeLearnAPI.getMetrics(),
              activeLearnAPI.getEpisodeHistory(),
            ]);

            setAutomatedStatus(status);
            setTrainingMetrics(metrics);
            setEpisodeHistory(history.episodes);
            setValidationAccuracy(metrics.best_val_acc || 0);

            // Handle training completion
            if (isRetraining && !status.is_training) {
              setIsRetraining(false);
              if (status.new_batch_available) {
                await handleStartNewBatch();
              }
            }
          } catch (error) {
            console.error("Failed to fetch status:", error);
          }
        },
        status?.is_training ? 1000 : 5000
      );
    }
    return () => clearInterval(interval);
  }, [isInitialized, isRetraining]);

  // Complete handleStartProject function for annotation.jsx
  const handleStartProject = async () => {
    if (!projectName || labels.length === 0) {
      setImageLoadError("Please set project name and labels");
      return;
    }

    // Check if we have files (handle both array and object formats)
    const hasFiles =
      (Array.isArray(loadedImages) && loadedImages.length > 0) ||
      (loadedImages &&
        typeof loadedImages === "object" &&
        loadedImages.fileCount > 0) ||
      (fileDataRef.current && fileDataRef.current.csvFile);

    if (!hasFiles) {
      setImageLoadError("Please upload images or a CSV file with image paths");
      return;
    }

    try {
      setIsLoading(true);
      setImageLoadError(null);

      // Only initialize project if not already initialized (avoids 422 error)
      if (!isInitialized) {
        console.log("Initializing new project...");

        const initResult = await activeLearnAPI.initializeProject({
          project_name: projectName,
          model_type: selectedModel,
          num_classes: labels.length,
          val_split: valSplit,
          initial_labeled_ratio: initialLabeledRatio,
          sampling_strategy: samplingStrategy,
          batch_size: parseInt(batchSize),
          epochs: parseInt(epochs),
          learning_rate: lrConfig.initial_lr,
        });

        console.log("Project initialized:", initResult);
        setIsInitialized(true);
      } else {
        console.log("Project already initialized, processing data only...");
      }

      // Handle CSV with labels using the ref
      if (fileDataRef.current?.type === "csv-with-labels") {
        setImageLoadError(
          `Processing CSV with annotations (using column: ${fileDataRef.current.labelColumn})...`
        );

        // Get the actual File object from the ref
        const csvFile = fileDataRef.current.csvFile;

        console.log("Processing CSV file:", {
          name: csvFile?.name,
          type: csvFile?.type,
          size: csvFile?.size,
          isFile: csvFile instanceof File,
        });

        if (!csvFile || !(csvFile instanceof File)) {
          throw new Error("CSV file object is not valid");
        }

        const result = await activeLearnAPI.uploadCSVPathsWithLabels(
          csvFile,
          fileDataRef.current.labelColumn,
          fileDataRef.current.delimiter || ",",
          valSplit,
          initialLabeledRatio
        );

        console.log("CSV with annotations processed:", result);

        // Update labels from label mapping
        if (result.label_mapping) {
          const existingLabels = [...labels];
          const newLabels = [...existingLabels];

          Object.keys(result.label_mapping).forEach((labelText) => {
            const labelIndex = result.label_mapping[labelText];
            while (newLabels.length <= labelIndex) {
              newLabels.push("");
            }
            if (!newLabels[labelIndex] || newLabels[labelIndex] === "") {
              newLabels[labelIndex] = labelText;
            }
          });

          const finalLabels = newLabels.filter((label) => label);
          if (finalLabels.length > 0) {
            setLabels(finalLabels);
            console.log("Updated labels from CSV:", finalLabels);
          }
        }

        setImageLoadError(
          `Successfully loaded ${result.stats.total} images (${result.stats.labeled} with labels, ${result.stats.unlabeled} unlabeled, ${result.stats.validation} for validation)`
        );

        // Start training if we have enough labeled data
        if (result.stats.labeled > 0) {
          setImageLoadError("Starting initial training with annotated data...");
          try {
            const episodeResult = await activeLearnAPI.startEpisode(
              epochs,
              batchSize
            );
            console.log("Initial training result:", episodeResult);
            if (episodeResult.final_val_acc) {
              setValidationAccuracy(episodeResult.final_val_acc);
            }
            setImageLoadError(
              "Initial training complete - Starting active learning"
            );
          } catch (error) {
            console.error("Initial training error:", error);
            setImageLoadError("Initial training error: " + error.message);
          }
        }

        // Get first batch for active learning
        try {
          await getNextBatch();
          setImageLoadError("Ready for active learning");
        } catch (batchError) {
          console.error("Error getting first batch:", batchError);
          setImageLoadError(
            "Warning: Could not get first batch - " + batchError.message
          );
        }
      }
      // Handle combined upload with labels
      else if (loadedImages.type === "combined-with-labels") {
        setImageLoadError(
          `Processing CSV with labels (${loadedImages.labelColumn}) and images together...`
        );
        const result = await activeLearnAPI.uploadCombinedWithLabels(
          fileDataRef.current.files,
          fileDataRef.current.labelColumn,
          valSplit,
          initialLabeledRatio
        );
        console.log("CSV with labels processed:", result);

        // Update labels from label mapping if needed
        if (result.label_mapping) {
          const newLabels = [...labels];
          Object.keys(result.label_mapping).forEach((labelText) => {
            const labelIndex = result.label_mapping[labelText];
            if (
              typeof newLabels[labelIndex] === "undefined" ||
              newLabels[labelIndex] === ""
            ) {
              while (newLabels.length <= labelIndex) {
                newLabels.push("");
              }
              newLabels[labelIndex] = labelText;
            }
          });

          if (newLabels.length > 0) {
            setLabels(newLabels);
          }
        }

        setImageLoadError(
          `Successfully loaded ${result.stats.total} images (${result.stats.labeled} with labels, ${result.stats.unlabeled} unlabeled, ${result.stats.validation} for validation)`
        );

        // Start training if we have enough labeled data
        if (result.stats.labeled > 0) {
          setImageLoadError("Starting initial training with annotated data...");
          try {
            const episodeResult = await activeLearnAPI.startEpisode(
              epochs,
              batchSize
            );
            console.log("Initial training result:", episodeResult);
            if (episodeResult.final_val_acc) {
              setValidationAccuracy(episodeResult.final_val_acc);
            }
            setImageLoadError("Initial training complete");
          } catch (error) {
            console.error("Initial training error:", error);
            setImageLoadError("Initial training error: " + error.message);
          }
        }

        // Get batch for active learning
        try {
          await getNextBatch();
          setImageLoadError("Ready for active learning");
        } catch (batchError) {
          console.error("Error getting batch:", batchError);
          setImageLoadError(
            "Warning: Could not get batch - " + batchError.message
          );
        }
      }
      // Handle combined upload without labels
      else if (loadedImages.type === "combined") {
        setImageLoadError("Processing CSV and image files together...");
        const result = await activeLearnAPI.uploadCombined(
          loadedImages.files,
          valSplit,
          initialLabeledRatio
        );
        console.log("Combined upload processed:", result);
        setImageLoadError(
          `Successfully processed ${result.split_info.total_images} images`
        );

        try {
          await getNextBatch();
          setImageLoadError("Ready for active learning");
        } catch (batchError) {
          console.error("Error getting batch:", batchError);
          setImageLoadError(
            "Warning: Could not get batch - " + batchError.message
          );
        }
      }
      // Handle CSV-only upload
      else if (loadedImages.type === "csv") {
        setImageLoadError("Processing CSV file with image paths...");

        const csvFile = loadedImages.file;
        const delimiter = loadedImages.delimiter || ",";

        try {
          const result = await activeLearnAPI.uploadCSVPaths(
            csvFile,
            delimiter,
            valSplit,
            initialLabeledRatio
          );
          console.log("CSV processing result:", result);
          setImageLoadError(
            `Successfully loaded ${result.split_info.total_images} images from CSV`
          );

          try {
            await getNextBatch();
            setImageLoadError("Ready for active learning");
          } catch (batchError) {
            console.error("Error getting batch:", batchError);
            setImageLoadError(
              "Warning: Could not get batch - " + batchError.message
            );
          }
        } catch (error) {
          console.error("CSV upload error:", error);
          setImageLoadError(
            `Error processing CSV: ${error.message}. Try uploading images directly or using the "Upload CSV + Images Together" option.`
          );
          return;
        }
      }
      // Handle regular image upload
      else if (Array.isArray(loadedImages) && loadedImages.length > 0) {
        setImageLoadError("Setting up initial dataset...");
        const result = await activeLearnAPI.uploadData(
          loadedImages,
          valSplit,
          initialLabeledRatio
        );
        console.log("Data split result:", result);

        try {
          await getNextBatch();
          setImageLoadError("Ready for active learning");
        } catch (batchError) {
          console.error("Error getting batch:", batchError);
          setImageLoadError(
            "Warning: Could not get batch - " + batchError.message
          );
        }
      }
    } catch (error) {
      console.error("Project initialization error:", error);

      // Handle specific error types
      if (
        error.message &&
        error.message.includes(
          "Could not find or process any images from the CSV"
        )
      ) {
        setImageLoadError(
          `Image file path issue: The system couldn't find the image files referenced in your CSV. 
This usually happens when the paths in the CSV don't match where images are stored on your system.

Possible solutions:
1. Use absolute paths in your CSV
2. Place images in the same folder as your application
3. Update your CSV with correct relative paths
4. Try using just filenames (without directory paths) in your CSV`
        );
      } else if (error.message && error.message.includes("422")) {
        setImageLoadError(
          "Project initialization failed. Try reloading the page and starting fresh."
        );
      } else {
        setImageLoadError("Error: " + error.message);
      }

      // Don't reset isInitialized if it was already true (pretrained model case)
      if (!isInitialized) {
        setIsInitialized(false);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const status = await activeLearnAPI.getStatus();
          setStatus(status);
        } catch (error) {
          console.error("Failed to fetch status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  useEffect(() => {
    let interval;
    if (isInitialized) {
      interval = setInterval(async () => {
        try {
          const status = await activeLearnAPI.getAutomatedTrainingStatus();
          setAutomatedStatus(status);
        } catch (error) {
          console.error("Failed to fetch automated training status:", error);
        }
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isInitialized]);

  const handleStartAutomatedTraining = async () => {
    try {
      // Check if project is initialized first
      if (!isInitialized) {
        setImageLoadError(
          "Please initialize the project with Start Project before starting automated training."
        );
        return;
      }

      // Check if images are loaded
      if (currentBatch.length === 0) {
        setImageLoadError(
          "No images loaded. Please ensure images are loaded before starting training."
        );
        try {
          await getNextBatch();
        } catch (error) {
          console.error("Failed to get batch before training:", error);
          return;
        }
      }

      // Make sure these values exist and are the correct type
      console.log("Starting automated training with:", {
        epochs,
        batchSize,
        samplingStrategy,
      });

      if (!epochs || !batchSize || !samplingStrategy) {
        setImageLoadError("Please set all training parameters");
        return;
      }

      // Ensure values are numbers
      const config = {
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        sampling_strategy: samplingStrategy,
        lr_config: lrConfig, // Add LR config
      };

      console.log("Sending config to API:", config);
      const response = await activeLearnAPI.startAutomatedTraining(config);
      console.log("API response:", response);

      if (response.status === "success") {
        setImageLoadError("Automated training started successfully");
        setIsRetraining(true);
      } else {
        setImageLoadError("Failed to start training: " + response.message);
      }
    } catch (error) {
      console.error("Start training error:", error);
      setImageLoadError("Failed to start automated training: " + error.message);
    }
  };

  const handleStopAutomatedTraining = async () => {
    try {
      await activeLearnAPI.stopAutomatedTraining();
      setImageLoadError("Automated training stopped");
    } catch (error) {
      setImageLoadError(error.message);
    }
  };

  const hasLoadedFiles = () => {
    // Check if we have files in any format
    if (Array.isArray(loadedImages) && loadedImages.length > 0) {
      return true; // Regular file upload
    }

    if (loadedImages && typeof loadedImages === "object") {
      // CSV uploads or other special formats
      if (
        loadedImages.type === "csv-with-labels" &&
        loadedImages.fileCount > 0
      ) {
        return true;
      }
      if (
        loadedImages.type === "combined-with-labels" &&
        loadedImages.fileCount > 0
      ) {
        return true;
      }
      if (loadedImages.type === "combined" && loadedImages.fileCount > 0) {
        return true;
      }
      if (loadedImages.type === "csv" && loadedImages.fileCount > 0) {
        return true;
      }
    }

    // Check if we have file data in the ref (CSV uploads)
    if (fileDataRef.current && fileDataRef.current.csvFile) {
      return true;
    }

    return false;
  };

  console.log("loadedImages:", loadedImages);
  console.log("loadedImages type:", typeof loadedImages);
  console.log("loadedImages.length:", loadedImages?.length);
  console.log("loadedImages.fileCount:", loadedImages?.fileCount);

  const handleSaveCheckpoint = async () => {
    try {
      setImageLoadError("Saving checkpoint...");
      const result = await activeLearnAPI.saveCheckpoint();
      setImageLoadError("Checkpoint saved successfully!");
      // Refresh checkpoint list
      const updatedCheckpoints = await activeLearnAPI.listCheckpoints();
      setCheckpoints(updatedCheckpoints.checkpoints);
    } catch (error) {
      setImageLoadError("Failed to save checkpoint: " + error.message);
    }
  };

  const handleLoadCheckpoint = async (checkpointId) => {
    try {
      setImageLoadError("Loading checkpoint...");
      await activeLearnAPI.loadCheckpoint(checkpointId);
      // Refresh current state
      const [status, metrics, history] = await Promise.all([
        activeLearnAPI.getStatus(),
        activeLearnAPI.getMetrics(),
        activeLearnAPI.getEpisodeHistory(),
      ]);
      setStatus(status);
      setMetrics(metrics);
      setEpisodeHistory(history.episodes);
      setImageLoadError("Checkpoint loaded successfully!");
    } catch (error) {
      setImageLoadError("Failed to load checkpoint: " + error.message);
    }
  };

  useEffect(() => {
    if (isInitialized) {
      const loadCheckpoints = async () => {
        try {
          const result = await activeLearnAPI.listCheckpoints();
          setCheckpoints(result.checkpoints);
        } catch (error) {
          console.error("Failed to load checkpoints:", error);
        }
      };
      loadCheckpoints();
    }
  }, [isInitialized]);

  useEffect(() => {
    // When currentBatch changes (like after loading a new batch)
    if (currentBatch.length > 0) {
      console.log(`New batch loaded with ${currentBatch.length} images`);

      // Reset batch stats whenever a new batch is loaded
      setBatchStats({
        totalImages: currentBatch.length,
        completed: 0,
        remaining: currentBatch.length,
        accuracy: validationAccuracy || 0.85,
        timeElapsed: "00:00",
      });

      // Reset batch progress state
      setIsBatchInProgress(true);
      setCurrentImageIndex(0);
    }
  }, [currentBatch]);

  return (
    <div className="container mx-auto p-6">
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column - Project Configuration */}
        <div>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-6">
            <TabsList className="grid grid-cols-2">
              <TabsTrigger value="new">New Project</TabsTrigger>
              <TabsTrigger value="import">Import Pretrained</TabsTrigger>
            </TabsList>
            <TabsContent value="new">
              <Card>
                <CardHeader>
                  <CardTitle>Project Configuration</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <Label htmlFor="project-name">Project Name</Label>
                      <Input
                        id="project-name"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        className="mt-1"
                        disabled={isInitialized}
                        placeholder="Enter project name"
                      />
                    </div>

                    <div>
                      <Label>Model Selection</Label>
                      <Select
                        value={selectedModel}
                        onValueChange={setSelectedModel}
                        disabled={isInitialized}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="resnet18">ResNet18</SelectItem>
                          <SelectItem value="resnet50">ResNet50</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Label Management */}
                    <div>
                      <Label>Labels</Label>
                      {labels.map((label, index) => (
                        <div key={index} className="flex gap-2 mt-2">
                          <Input
                            value={label}
                            onChange={(e) =>
                              handleLabelChange(index, e.target.value)
                            }
                            placeholder={`Label ${index + 1}`}
                          />
                          {labels.length > 1 && (
                            <Button
                              variant="outline"
                              size="icon"
                              onClick={() => handleRemoveLabel(index)}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      ))}
                      <Button
                        variant="outline"
                        onClick={handleAddLabel}
                        className="w-full mt-2"
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Add Label
                      </Button>
                    </div>

                    {/* Sampling Strategy Controls */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Sampling Strategy</Label>
                        <Select
                          value={samplingStrategy}
                          onValueChange={setSamplingStrategy}
                          disabled={batchStats.completed > 0} // Only disable during training or mid-batch
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select strategy" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="least_confidence">
                              Least Confidence
                            </SelectItem>
                            <SelectItem value="margin">
                              Margin Sampling
                            </SelectItem>
                            <SelectItem value="entropy">Entropy</SelectItem>
                            <SelectItem value="diversity">
                              Diversity-based
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label>Learning Rate Strategy</Label>
                        <Select
                          value={lrConfig.strategy}
                          onValueChange={(value) =>
                            setLrConfig((prev) => ({
                              ...prev,
                              strategy: value,
                            }))
                          }
                          disabled={isRetraining}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select LR strategy" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="plateau">
                              Reduce on Plateau
                            </SelectItem>
                            <SelectItem value="cosine">
                              Cosine Annealing
                            </SelectItem>
                            <SelectItem value="warmup">
                              One Cycle with Warmup
                            </SelectItem>
                            <SelectItem value="step">Step Decay</SelectItem>
                          </SelectContent>
                        </Select>

                        <Label>Initial Learning Rate</Label>
                        <Input
                          type="number"
                          value={lrConfig.initial_lr}
                          onChange={(e) =>
                            setLrConfig((prev) => ({
                              ...prev,
                              initial_lr: parseFloat(e.target.value),
                            }))
                          }
                          min={0.0001}
                          max={0.1}
                          step={0.0001}
                          disabled={isRetraining}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Batch Size</Label>
                        <Input
                          type="number"
                          value={batchSize}
                          onChange={(e) => {
                            const newBatchSize = Number(e.target.value);
                            setBatchSize(newBatchSize);
                            setBatchStats((prev) => ({
                              ...prev,
                              totalImages: newBatchSize,
                              remaining: newBatchSize,
                            }));
                          }}
                          min={1}
                          max={100}
                          disabled={isRetraining || batchStats.completed > 0} // Only disable during training or mid-batch
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Epochs</Label>
                        <Input
                          type="number"
                          value={epochs}
                          onChange={(e) => {
                            const newEpochs = Number(e.target.value);
                            setEpochs(newEpochs);
                          }}
                          min={10}
                          disabled={isRetraining} // Only disable during training
                        />
                      </div>
                    </div>

                    {/* Data Split Configuration */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Validation Split</Label>
                        <div className="flex items-center space-x-2">
                          <Input
                            type="range"
                            min="0.1"
                            max="0.3"
                            step="0.05"
                            value={valSplit}
                            onChange={(e) =>
                              setValSplit(parseFloat(e.target.value))
                            }
                            className="flex-1"
                          />
                          <span className="text-sm w-16 text-right">
                            {(valSplit * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label>Initial Labeled Ratio</Label>
                        <div className="flex items-center space-x-2">
                          <Input
                            type="range"
                            min="0.1"
                            max="0.8"
                            step="0.05"
                            value={initialLabeledRatio}
                            onChange={(e) =>
                              setInitialLabeledRatio(parseFloat(e.target.value))
                            }
                            className="flex-1"
                          />
                          <span className="text-sm w-16 text-right">
                            {(initialLabeledRatio * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {loadedImages.length > 0 && (
                      <div className="text-sm text-gray-600">
                        Loaded {loadedImages.length} images
                      </div>
                    )}

                    {loadedImages.length > 0 && (
                      <Card className="bg-secondary/50">
                        <CardContent className="pt-4">
                          <h4 className="font-medium mb-2">
                            Data Split Preview
                          </h4>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span>Total Images:</span>
                              <span>{loadedImages.length}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Validation Set:</span>
                              <span>
                                {Math.round(loadedImages.length * valSplit)}{" "}
                                images
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Initial Labeled Set:</span>
                              <span>
                                {Math.round(
                                  loadedImages.length *
                                    (1 - valSplit) *
                                    initialLabeledRatio
                                )}{" "}
                                images
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Remaining Unlabeled:</span>
                              <span>
                                {loadedImages.length -
                                  Math.round(loadedImages.length * valSplit) -
                                  Math.round(
                                    loadedImages.length *
                                      (1 - valSplit) *
                                      initialLabeledRatio
                                  )}{" "}
                                images
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    <ImageLoader
                      onImagesLoaded={handleImagesLoaded}
                      onError={setImageLoadError}
                    />

                    {imageLoadError && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{imageLoadError}</AlertDescription>
                      </Alert>
                    )}
                    <Button
                      className="w-full"
                      onClick={handleStartProject}
                      disabled={
                        !projectName ||
                        labels.length === 0 ||
                        !(
                          (Array.isArray(loadedImages) &&
                            loadedImages.length > 0) ||
                          (loadedImages &&
                            typeof loadedImages === "object" &&
                            loadedImages.fileCount > 0)
                        ) ||
                        !isInitialized ||
                        isLoading
                      }
                    >
                      {isLoading ? "Initializing Project..." : "Start Project"}
                    </Button>
                    {isInitialized && (
                      <AutomatedTrainingControls
                        onStart={handleStartAutomatedTraining}
                        onStop={handleStopAutomatedTraining}
                        status={automatedStatus}
                        metrics={trainingMetrics} // Pass metrics to controls
                        disabled={!isInitialized}
                        episode_history={episodeHistory}
                      />
                    )}
                    <div>
                      <ActiveLearningStatus
                        status={status}
                        onStartNewBatch={handleStartNewBatch}
                      />
                    </div>
                    <CheckpointControls
                      onSave={handleSaveCheckpoint}
                      onLoad={handleLoadCheckpoint}
                      checkpoints={checkpoints}
                    />
                    <div className="space-y-4">
                      <div className="flex gap-4">
                        <Button
                          onClick={async () => {
                            try {
                              const blob = await activeLearnAPI.exportModel();
                              const url = window.URL.createObjectURL(blob);
                              const a = document.createElement("a");
                              a.href = url;
                              // Update export filename with parameters
                              const filename = `${projectName}_${samplingStrategy}_e${epochs}_b${batchSize}_model.pt`;
                              a.download = filename;
                              document.body.appendChild(a);
                              a.click();
                              window.URL.revokeObjectURL(url);
                            } catch (error) {
                              setImageLoadError(
                                "Failed to export model: " + error.message
                              );
                            }
                          }}
                          disabled={isBatchInProgress}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Export Model
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="import">
              <div className="space-y-6">
                <PretrainedModelImport
                  onImportSuccess={(result) => {
                    setProjectName(result.project_name);
                    setSelectedModel(result.model_type);
                    setIsInitialized(true);
                    setImageLoadError(
                      "Pre-trained model imported successfully!"
                    );

                    // Fetch any needed state updates
                    activeLearnAPI.getStatus().then((status) => {
                      setStatus(status);
                    });

                    activeLearnAPI.getMetrics().then((metrics) => {
                      setMetrics(metrics);
                    });
                  }}
                  onError={(errorMsg) => {
                    setImageLoadError(errorMsg);
                  }}
                />

                {isInitialized && (
                  <ModelAdaptationControls
                    onAdaptSuccess={(result) => {
                      setImageLoadError(
                        `Model adaptation successful using ${result.adaptation_type} strategy`
                      );
                    }}
                    onError={(errorMsg) => {
                      setImageLoadError(`Adaptation error: ${errorMsg}`);
                    }}
                    disabled={!isInitialized || isRetraining}
                  />
                )}
              </div>
            </TabsContent>
          </Tabs>
          {/* In your metrics section */}
          {(isRetraining || metrics?.episodeAccuracies?.length > 0) && (
            <EnhancedMetricsDisplay
              metrics={metrics}
              episode_history={episodeHistory}
              lr_history={lrHistory}
            />
          )}
        </div>

        {/* Right Column - Image Annotation */}
        <div>
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle>Image Annotation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Image Display with Navigation */}
                <div className="bg-gray-100 rounded-lg overflow-hidden">
                  <img
                    src={currentImage}
                    alt="Current image for annotation"
                    className="w-full object-contain max-h-96"
                  />
                  {loadedImages.length > 0 && (
                    <div className="p-4 bg-white border-t flex justify-between items-center">
                      <Button
                        onClick={handlePreviousImage}
                        disabled={currentImageIndex === 0}
                        variant="outline"
                      >
                        Previous
                      </Button>
                      <span className="text-sm text-gray-600">
                        Image {currentImageIndex + 1} of {batchSize}
                      </span>
                      <Button
                        onClick={handleNextImage}
                        disabled={currentImageIndex === loadedImages.length - 1}
                        variant="outline"
                      >
                        Next
                      </Button>
                    </div>
                  )}
                </div>
                {imageLoadError && (
                  <Alert
                    variant={
                      imageLoadError.startsWith("Training") ||
                      imageLoadError.startsWith("Getting")
                        ? "default"
                        : "destructive"
                    }
                  >
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{imageLoadError}</AlertDescription>
                  </Alert>
                )}
                {/* Label Selection */}
                <div className="space-y-2">
                  <Label>Assign Label</Label>
                  <RadioGroup
                    value={selectedLabel}
                    onValueChange={setSelectedLabel}
                    className="space-y-2"
                  >
                    {labels.map((label, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <RadioGroupItem
                          value={index.toString()}
                          id={`label-${index}`}
                        />
                        <Label htmlFor={`label-${index}`}>{label}</Label>
                      </div>
                    ))}
                  </RadioGroup>
                </div>
                {/* Action Buttons */}
                <div className="flex justify-between pt-4">
                  <Button
                    className="w-full"
                    onClick={handleSubmitLabel}
                    disabled={!selectedLabel}
                  >
                    Submit Label
                  </Button>
                </div>

                <BatchProgress
                  currentBatch={currentBatch}
                  batchStats={batchStats}
                  onSubmitLabel={handleSubmitLabel}
                  selectedLabel={selectedLabel}
                  isRetraining={isRetraining}
                  validationAccuracy={validationAccuracy}
                />

                <ModelPredictions
                  predictions={
                    currentBatch[currentImageIndex]?.predictions || []
                  }
                  labels={labels}
                />

                {isRetraining && (
                  <div className="mt-4">
                    <Card className="bg-blue-50 border-blue-200">
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                          <p className="text-blue-700">
                            Training in progress... Please wait
                          </p>
                        </div>
                        <p className="text-sm text-blue-600 mt-2">
                          This may take several minutes depending on the dataset
                          size
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ActiveLearningUI;
