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
import { Upload, CheckCircle, AlertCircle } from "lucide-react";
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
  const [verified, setVerified] = useState(false);

  const modelFileRef = useRef(null);

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

        // Handle different model types differently
        if (result.detected_type === "vision-transformer") {
          setModelType("vision-transformer");
          if (!result.num_classes || result.num_classes > 100) {
            setNumClasses(numClasses > 1 ? numClasses : 2);
          } else {
            setNumClasses(result.num_classes);
          }
        } else {
          // For ResNet and other models, use detected values
          if (result.model_type) setModelType(result.model_type);
          if (result.num_classes) setNumClasses(result.num_classes);
        }

        setImportStatus(`Model verified: ${result.message}`);
        setVerified(true);
      } else {
        setImportError(`Incompatible model: ${result.message}`);
        setImportStatus(null);
      }
    } catch (error) {
      setImportError(`Failed to verify model: ${error.message}`);
      setImportStatus(null);
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

      // Import the pretrained model with user-specified parameters
      const importResult = await activeLearnAPI.importPretrainedModel(
        selectedFile,
        modelType,
        numClasses, // This is the user-specified value
        projectName
      );

      console.log("Model import result:", importResult);
      setImportStatus("Model imported successfully");
      setModelImported(true);

      // Pass the user-specified numClasses, not the detected one
      const resultWithUserClasses = {
        ...importResult,
        num_classes: numClasses, // Override with user-specified value
        user_specified_classes: numClasses,
      };

      // Call the success callback with the corrected result
      onImportSuccess(resultWithUserClasses);
    } catch (error) {
      console.error("Import error:", error);
      setImportError(`Failed to import model: ${error.message}`);
      setImportStatus(null);
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
              <SelectItem value="vision-transformer">
                Vision Transformer
              </SelectItem>
              <SelectItem value="custom">Custom Model</SelectItem>
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
          disabled={!selectedFile || isImporting || !verified}
        >
          {isImporting ? "Importing..." : "Import Model"}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default PretrainedModelImport;
