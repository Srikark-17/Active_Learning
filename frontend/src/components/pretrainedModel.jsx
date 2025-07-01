import React, { useState, useRef } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "./ui/card";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Upload, Database, CheckCircle, AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Alert, AlertDescription } from "./ui/alert";
import activeLearnAPI from "../services/activelearning";

const PretrainedModelImport = ({ onImportSuccess, onError }) => {
  const [modelType, setModelType] = useState("resnet50");
  const [numClasses, setNumClasses] = useState(2);
  const [projectName, setProjectName] = useState("pretrained_project");
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState(null);
  const [importStatus, setImportStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [labelColumn, setLabelColumn] = useState("annotation");
  const [delimiter, setDelimiter] = useState(",");
  const [modelImported, setModelImported] = useState(false);

  const modelFileRef = useRef(null);
  const csvFileRef = useRef(null);

  const handleModelFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setImportError(null);
    setImportStatus("Verifying model compatibility...");

    try {
      // Verify model compatibility
      const result = await activeLearnAPI.verifyModelCompatibility(file);

      if (result.compatible) {
        setModelInfo(result);
        // Update UI with detected info
        if (result.model_type) setModelType(result.model_type);
        if (result.num_classes) setNumClasses(result.num_classes);

        setImportStatus(`Model verified: ${result.message}`);
      } else {
        setImportError(`Incompatible model: ${result.message}`);
        setImportStatus(null);
      }
    } catch (error) {
      setImportError(`Failed to verify model: ${error.message}`);
      setImportStatus(null);
    }
  };

  const handleCsvFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setCsvFile(file);
    setImportError(null);

    // Try to detect if this is a CSV file with annotations
    if (file.name.endsWith(".csv") || file.name.endsWith(".tsv")) {
      // Read file to auto-detect columns
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          const lines = content.split("\n");

          if (lines.length > 0) {
            // Try to detect delimiter
            const firstLine = lines[0];
            let detectedDelimiter = ",";

            if (firstLine.includes("\t")) {
              detectedDelimiter = "\t";
            } else if (firstLine.includes(";")) {
              detectedDelimiter = ";";
            }

            setDelimiter(detectedDelimiter);

            // Parse header to find annotation column
            const headers = firstLine
              .split(detectedDelimiter)
              .map((h) => h.trim());

            // Look for annotation column
            const annotationColumns = [
              "annotation",
              "label",
              "class",
              "category",
            ];
            let detectedColumn = null;

            for (const col of annotationColumns) {
              const found = headers.find((h) => h.toLowerCase() === col);
              if (found) {
                detectedColumn = found;
                break;
              }
            }

            if (detectedColumn) {
              setLabelColumn(detectedColumn);
              setImportStatus(
                `CSV detected with annotations column: ${detectedColumn}`
              );
            } else {
              setImportStatus(`CSV loaded, but no annotation column detected`);
            }
          }
        } catch (error) {
          console.error("Error parsing CSV:", error);
        }
      };

      reader.readAsText(file);
    }
  };

  const handleImportModel = async () => {
    if (!selectedFile) {
      setImportError("Please select a model file");
      return;
    }

    try {
      setIsImporting(true);
      setImportError(null);
      setImportStatus("Importing model...");

      // Import the pretrained model
      const importResult = await activeLearnAPI.importPretrainedModel(
        selectedFile,
        modelType,
        numClasses,
        projectName
      );

      console.log("Model import result:", importResult);
      setImportStatus("Model imported successfully");
      setModelImported(true);

      // If CSV file is selected, process it with annotations
      if (csvFile) {
        setImportStatus("Processing CSV with annotations...");

        try {
          const csvResult = await activeLearnAPI.uploadCSVPathsWithLabels(
            csvFile,
            labelColumn,
            delimiter,
            0.2, // Default validation split
            0.4 // Default initial labeled ratio
          );

          console.log("CSV processing result:", csvResult);

          setImportStatus(
            `Model imported and CSV processed. ${csvResult.stats.labeled} labeled images, ${csvResult.stats.unlabeled} unlabeled images.`
          );

          // If we have labeled data, start training
          if (csvResult.stats.labeled > 0) {
            setImportStatus("Starting initial training with annotated data...");

            try {
              const trainingResult = await activeLearnAPI.startEpisode(
                10, // Default epochs
                32 // Default batch size
              );

              console.log("Initial training result:", trainingResult);
              setImportStatus(
                "Initial training complete. Model ready for active learning."
              );

              // Get first batch for active learning
              await activeLearnAPI.getNextBatch();
            } catch (error) {
              console.error("Initial training error:", error);
              setImportError(`Initial training error: ${error.message}`);
              // Still consider import successful even if training fails
            }
          }
        } catch (error) {
          console.error("CSV processing error:", error);
          setImportError(`CSV processing error: ${error.message}`);
          // Consider model import successful even if CSV processing fails
        }
      }

      // Call the success callback with import result
      onImportSuccess(importResult);
    } catch (error) {
      console.error("Import error:", error);
      setImportError(`Failed to import model: ${error.message}`);
      setImportStatus(null);
    } finally {
      setIsImporting(false);
    }
  };

  const handleAdaptModel = async (adaptationType = "last_layer") => {
    if (!modelImported) {
      setImportError("Please import a model first");
      return;
    }

    try {
      setIsImporting(true);
      setImportError(null);
      setImportStatus(`Adapting model (${adaptationType})...`);

      const adaptResult = await activeLearnAPI.adaptPretrainedModel(
        true, // freeze layers
        adaptationType
      );

      console.log("Model adaptation result:", adaptResult);
      setImportStatus(
        `Model adapted successfully using ${adaptationType} strategy`
      );
    } catch (error) {
      console.error("Adaptation error:", error);
      setImportError(`Failed to adapt model: ${error.message}`);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Import Pretrained Model</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="project-name">Project Name</Label>
          <Input
            id="project-name"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            disabled={isImporting}
          />
        </div>

        <div className="space-y-2">
          <Label>Model Type</Label>
          <Select
            value={modelType}
            onValueChange={setModelType}
            disabled={isImporting}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select model type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="resnet18">ResNet18</SelectItem>
              <SelectItem value="resnet50">ResNet50</SelectItem>
              <SelectItem value="custom">Custom / Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="num-classes">Number of Classes</Label>
          <Input
            id="num-classes"
            type="number"
            min="2"
            value={numClasses}
            onChange={(e) => setNumClasses(parseInt(e.target.value))}
            disabled={isImporting}
          />
        </div>

        <div className="space-y-2">
          <Label>Import Model File</Label>
          <div className="flex gap-2">
            <Button
              onClick={() => modelFileRef.current.click()}
              variant="outline"
              className="w-full"
              disabled={isImporting}
            >
              <Upload className="h-4 w-4 mr-2" />
              Select Model File (.pt)
            </Button>
            <input
              type="file"
              ref={modelFileRef}
              onChange={handleModelFileSelect}
              className="hidden"
              accept=".pt,.pth,.ckpt"
            />
          </div>
          {selectedFile && (
            <div className="text-sm text-gray-600">
              Selected: {selectedFile.name}
            </div>
          )}
          {modelInfo && (
            <Alert className="bg-blue-50 border-blue-200">
              <CheckCircle className="h-4 w-4 text-blue-500" />
              <AlertDescription className="text-blue-700">
                Detected model type: {modelInfo.model_type}, Classes:{" "}
                {modelInfo.num_classes || "Unknown"}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <div className="space-y-2">
          <Label>Optional: Import CSV with Annotations</Label>
          <div className="flex gap-2">
            <Button
              onClick={() => csvFileRef.current.click()}
              variant="outline"
              className="w-full"
              disabled={isImporting}
            >
              <Database className="h-4 w-4 mr-2" />
              Select CSV File
            </Button>
            <input
              type="file"
              ref={csvFileRef}
              onChange={handleCsvFileSelect}
              className="hidden"
              accept=".csv,.tsv,.txt"
            />
          </div>
          {csvFile && (
            <div className="text-sm text-gray-600">
              Selected: {csvFile.name}
            </div>
          )}
          {csvFile && (
            <div className="space-y-2">
              <div className="flex gap-2">
                <div className="flex-1">
                  <Label>Label Column</Label>
                  <Input
                    value={labelColumn}
                    onChange={(e) => setLabelColumn(e.target.value)}
                    disabled={isImporting}
                    placeholder="annotation"
                  />
                </div>
                <div className="flex-1">
                  <Label>Delimiter</Label>
                  <Select
                    value={delimiter}
                    onValueChange={setDelimiter}
                    disabled={isImporting}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select delimiter" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value=",">Comma (,)</SelectItem>
                      <SelectItem value="\t">Tab</SelectItem>
                      <SelectItem value=";">Semicolon (;)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          )}
        </div>

        {importError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{importError}</AlertDescription>
          </Alert>
        )}

        {importStatus && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>{importStatus}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex flex-col gap-2">
        <Button
          className="w-full"
          onClick={handleImportModel}
          disabled={!selectedFile || isImporting}
        >
          {isImporting ? "Importing..." : "Import Model"}
        </Button>

        {modelImported && (
          <div className="w-full flex gap-2">
            <Button
              className="flex-1"
              variant="outline"
              onClick={() => handleAdaptModel("last_layer")}
              disabled={isImporting}
            >
              Fine-tune Last Layer
            </Button>
            <Button
              className="flex-1"
              variant="outline"
              onClick={() => handleAdaptModel("full_finetune")}
              disabled={isImporting}
            >
              Full Fine-tuning
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};

export default PretrainedModelImport;
