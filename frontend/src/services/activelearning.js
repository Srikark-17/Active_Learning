const API_URL = "http://localhost:8000";

const activeLearnAPI = {
  async initializeProject({
    project_name,
    model_type,
    num_classes,
    val_split,
    initial_labeled_ratio,
    sampling_strategy,
    batch_size,
    epochs,
    learning_rate,
  }) {
    const response = await fetch(`${API_URL}/init`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_name,
        model_type,
        num_classes,
        val_split,
        initial_labeled_ratio,
        sampling_strategy,
        batch_size,
        epochs,
        learning_rate,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(
        error.detail || `Failed to initialize project: ${response.statusText}`
      );
    }

    return await response.json();
  },

  async getBatch(strategy, batchSize) {
    const response = await fetch(`${API_URL}/get-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy: strategy,
        batch_size: parseInt(batchSize),
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get batch");
    }
    return response.json();
  },

  async uploadData(files, valSplit, initialLabeledRatio) {
    const formData = new FormData();
    for (let file of files) {
      formData.append("files", file);
    }
    if (valSplit !== undefined) {
      formData.append("val_split", valSplit);
    }
    if (initialLabeledRatio !== undefined) {
      formData.append("initial_labeled_ratio", initialLabeledRatio);
    }

    const response = await fetch(`${API_URL}/upload-data`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to upload data: ${response.statusText}`);
    }

    return await response.json();
  },

  async submitLabel(imageId, label) {
    const response = await fetch(`${API_URL}/submit-label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: imageId,
        label: label,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to submit label");
    }
    const result = await response.json();
    return result; // Now includes batch_complete and is_training flags
  },

  async trainModel(epochs = 10, batchSize = 32, learningRate = 0.001) {
    const response = await fetch(`${API_URL}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      }),
    });
    return response.json();
  },

  async startEpisode(epochs = 10, batchSize = 32, learningRate = 0.001) {
    const response = await fetch(`${API_URL}/train-episode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start episode: ${response.statusText}`);
    }

    return await response.json();
  },

  async getStatus() {
    const response = await fetch(`${API_URL}/status`);
    return response.json();
  },

  async getMetrics() {
    const response = await fetch(`${API_URL}/metrics`);
    return response.json();
  },

  async getEpisodeHistory() {
    const response = await fetch(`${API_URL}/episode-history`);
    if (!response.ok) {
      throw new Error(`Failed to get episode history: ${response.statusText}`);
    }
    return await response.json();
  },

  getImageUrl(imageId) {
    return `${API_URL}/image/${imageId}`;
  },

  async exportModel() {
    const response = await fetch(`${API_URL}/export-model`, {
      method: "GET",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to export model");
    }
    return response.blob();
  },

  async importModel(file) {
    const formData = new FormData();
    formData.append("uploaded_file", file); // Match the parameter name in FastAPI

    const response = await fetch(`${API_URL}/import-model`, {
      method: "POST",
      body: formData, // Don't set Content-Type header, let browser set it with boundary
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to import model");
    }

    return response.json();
  },

  async getValidationStatus() {
    const response = await fetch(`${API_URL}/validation-status`);
    if (!response.ok) {
      throw new Error(
        `Failed to get validation status: ${response.statusText}`
      );
    }
    return await response.json();
  },

  async startAutomatedTraining(config) {
    const newConfig = {
      epochs: config.epochs,
      batch_size: config.batch_size,
      sampling_strategy: config.sampling_strategy,
      learning_rate: 0.001,
    };

    const response = await fetch(`${API_URL}/start-automated-training`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json", // This is important
      },
      body: JSON.stringify(newConfig),
    });

    if (!response.ok) {
      const error = await response.json();
      console.log(error);
      throw new Error(
        error.detail ||
          `Failed to start automated training: ${JSON.stringify(error)}`
      );
    }
    return response.json();
  },

  async stopAutomatedTraining() {
    const response = await fetch(`${API_URL}/stop-automated-training`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(
        `Failed to stop automated training: ${JSON.stringify(
          response.statusText
        )}`
      );
    }
    return response.json();
  },

  async getAutomatedTrainingStatus() {
    const response = await fetch(`${API_URL}/automated-training-status`);
    if (!response.ok) {
      throw new Error(
        `Failed to get automated training status: ${response.statusText}`
      );
    }
    return response.json();
  },

  async getNextBatch() {
    const response = await fetch(`${API_URL}/get-next-batch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (!response.ok) {
      throw new Error(`Failed to get next batch: ${response.statusText}`);
    }
    return response.json();
  },

  async resetTrainingState() {
    const response = await fetch(`${API_URL}/reset-training-state`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Failed to reset training state: ${response.statusText}`);
    }
    return response.json();
  },

  configureLRScheduler: async (config) => {
    const response = await fetch(`${API_URL}/configure-lr-scheduler`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return response.json();
  },

  getLRSchedulerStatus: async () => {
    try {
      const response = await fetch(`${API_URL}/lr-scheduler-status`);
      if (!response.ok) {
        console.warn("Failed to get LR scheduler status");
        return null;
      }
      return response.json();
    } catch (error) {
      console.error("Error getting LR scheduler status:", error);
      return null;
    }
  },

  saveCheckpoint: async () => {
    const response = await fetch(`${API_URL}/save-checkpoint`, {
      method: "POST",
    });
    return response.json();
  },

  loadCheckpoint: async (checkpointId) => {
    const response = await fetch(`${API_URL}/load-checkpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_path: checkpointId }),
    });
    return response.json();
  },

  listCheckpoints: async () => {
    const response = await fetch(`${API_URL}/list-checkpoints`);
    return response.json();
  },
};

export default activeLearnAPI;
