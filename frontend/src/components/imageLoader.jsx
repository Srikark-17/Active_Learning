import React, { useRef, useState } from "react";
import { Button } from "./ui/button";
import { Upload, FolderUp, Archive, Table } from "lucide-react";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import Papa from "papaparse";
import { Tooltip } from "react-tooltip";

const ImageLoader = ({ onImagesLoaded, onError }) => {
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const csvAndImagesRef = useRef(null);
  const csvWithLabelsRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [labelColumn, setLabelColumn] = useState("label");
  const [csvPreview, setCsvPreview] = useState(null);
  const [showLabelDialog, setShowLabelDialog] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [imageFiles, setImageFiles] = useState([]);
  const [detectedLabels, setDetectedLabels] = useState([]);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (files.length > 0) {
      onImagesLoaded(files);
    } else {
      onError("No valid image files selected");
    }
  };

  const handleFolderChange = (e) => {
    const files = Array.from(e.target.files).filter((file) =>
      file.type.startsWith("image/"),
    );
    if (files.length > 0) {
      onImagesLoaded(files);
    } else {
      onError("No valid image files in selected folder");
    }
  };

  const handleCsvAndImagesChange = async (e) => {
    setIsLoading(true);
    const files = Array.from(e.target.files);
    if (files.length === 0) {
      setIsLoading(false);
      return;
    }

    const csvFiles = files.filter(
      (f) =>
        f.name.endsWith(".csv") ||
        f.name.endsWith(".tsv") ||
        f.name.endsWith(".txt"),
    );
    const imgFiles = files.filter((f) => f.type.startsWith("image/"));

    if (csvFiles.length === 0) {
      onError("Please include a CSV file with your selection");
      setIsLoading(false);
      return;
    }
    if (imgFiles.length === 0) {
      onError("Please include image files with your selection");
      setIsLoading(false);
      return;
    }

    onImagesLoaded([...csvFiles, ...imgFiles], "combined");
    setIsLoading(false);
  };

  const handleCsvWithLabelsSelect = async (e) => {
    setIsLoading(true);
    const files = Array.from(e.target.files);
    if (files.length === 0) {
      setIsLoading(false);
      return;
    }

    const csvFiles = files.filter(
      (f) =>
        f.name.endsWith(".csv") ||
        f.name.endsWith(".tsv") ||
        f.name.endsWith(".txt"),
    );
    const imgFiles = files.filter((f) => f.type.startsWith("image/"));

    if (csvFiles.length === 0) {
      onError("Please include a CSV file with your selection");
      setIsLoading(false);
      return;
    }

    const selectedCsvFile = csvFiles[0];
    setCsvFile(selectedCsvFile);
    setImageFiles(imgFiles.length > 0 ? imgFiles : []);

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const content = ev.target.result;

        const labelColumnNames = [
          "annotation",
          "label",
          "class",
          "category",
          "target",
          "classification",
        ];

        Papa.parse(content, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            if (result.errors.length > 0) {
              onError(`Error parsing CSV: ${result.errors[0].message}`);
              setIsLoading(false);
              return;
            }

            const columns = result.meta.fields || [];

            let detectedLabelColumn = null;
            for (const col of labelColumnNames) {
              if (columns.includes(col)) {
                detectedLabelColumn = col;
                break;
              }
              const match = columns.find((c) => c.toLowerCase() === col);
              if (match) {
                detectedLabelColumn = match;
                break;
              }
            }
            if (!detectedLabelColumn && columns.length > 1) {
              detectedLabelColumn = columns[1];
            }

            if (detectedLabelColumn) setLabelColumn(detectedLabelColumn);

            const uniqueLabels = new Set();
            for (const row of result.data) {
              const val =
                detectedLabelColumn && row[detectedLabelColumn]
                  ? row[detectedLabelColumn].trim()
                  : null;
              if (val) uniqueLabels.add(val);
            }

            const allLabels = Array.from(uniqueLabels).filter(Boolean).sort();
            setDetectedLabels(allLabels);
            setCsvPreview(result.data.slice(0, 5));
            setShowLabelDialog(true);
            setIsLoading(false);
          },
        });
      } catch (error) {
        onError(`Error reading CSV: ${error.message}`);
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      onError("Failed to read the CSV file");
      setIsLoading(false);
    };
    reader.readAsText(selectedCsvFile);
  };

  const handleConfirmLabelColumn = () => {
    if (imageFiles.length === 0) {
      onImagesLoaded([csvFile], "csv-with-labels", labelColumn, detectedLabels);
    } else {
      onImagesLoaded(
        [csvFile, ...imageFiles],
        "combined-with-labels",
        labelColumn,
        detectedLabels,
      );
    }
    setShowLabelDialog(false);
  };

  return (
    <>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Button
            onClick={() => fileInputRef.current.click()}
            className="w-full"
            variant="outline"
            data-tooltip-id="upload-img-tooltip"
            data-tooltip-content="Upload images for active learning classification"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Images
          </Button>
          <Tooltip id="upload-img-tooltip" />
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            multiple
            className="hidden"
          />

          <Button
            onClick={() => folderInputRef.current.click()}
            className="w-full"
            variant="outline"
            data-tooltip-id="upload-folder-tooltip"
            data-tooltip-content="Upload a folder of images for classification"
          >
            <FolderUp className="h-4 w-4 mr-2" />
            Upload Folder
          </Button>
          <Tooltip id="upload-folder-tooltip" />
          <input
            type="file"
            ref={folderInputRef}
            onChange={handleFolderChange}
            webkitdirectory="true"
            directory="true"
            multiple
            className="hidden"
          />
        </div>

        <div className="grid grid-cols-1 gap-4">
          <Button
            onClick={() => csvAndImagesRef.current.click()}
            className="w-full"
            variant="outline"
            disabled={isLoading}
            data-tooltip-id="upload-csv-tooltip"
            data-tooltip-content="Upload a CSV with image paths for classification"
          >
            <Archive className="h-4 w-4 mr-2" />
            {isLoading ? "Processing..." : "Upload CSV + Images Together"}
          </Button>
          <Tooltip id="upload-csv-tooltip" />
          <input
            type="file"
            ref={csvAndImagesRef}
            onChange={handleCsvAndImagesChange}
            accept=".csv,.tsv,.txt,image/*"
            multiple
            className="hidden"
          />

          <Button
            onClick={() => csvWithLabelsRef.current.click()}
            className="w-full"
            variant="default"
            disabled={isLoading}
            data-tooltip-id="upload-csv-labels-tooltip"
            data-tooltip-content="Upload a CSV with image paths and labels for classification"
          >
            <Table className="h-4 w-4 mr-2" />
            {isLoading ? "Processing..." : "Upload CSV with Labels + Images"}
          </Button>
          <Tooltip id="upload-csv-labels-tooltip" />
          <input
            type="file"
            ref={csvWithLabelsRef}
            onChange={handleCsvWithLabelsSelect}
            accept=".csv,.tsv,.txt,image/*"
            multiple
            className="hidden"
          />
        </div>

        {isLoading && (
          <div className="text-center text-sm text-gray-500">
            Processing files...
          </div>
        )}
      </div>

      <Dialog open={showLabelDialog} onOpenChange={setShowLabelDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Label Column</DialogTitle>
            <DialogDescription>
              We detected the following columns in your CSV. Please confirm
              which column contains the class labels:
            </DialogDescription>
          </DialogHeader>

          <div className="py-4">
            <Label htmlFor="label-column">Label Column</Label>
            <Input
              id="label-column"
              value={labelColumn}
              onChange={(e) => setLabelColumn(e.target.value)}
              className="mt-1"
              placeholder="Enter label column name"
            />
            {detectedLabels.length > 0 && (
              <p className="mt-2 text-sm text-gray-500">
                Detected classes ({detectedLabels.length}):{" "}
                {detectedLabels.join(", ")}
              </p>
            )}
          </div>

          {csvPreview && csvPreview.length > 0 && (
            <div className="max-h-60 overflow-y-auto">
              <h4 className="font-medium mb-2">CSV Preview (first 5 rows)</h4>
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(csvPreview[0]).map((column) => (
                      <th
                        key={column}
                        className={`px-3 py-2 text-left font-medium text-gray-500 ${column === labelColumn ? "bg-blue-100" : ""}`}
                      >
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                  {csvPreview.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {Object.keys(row).map((column, colIndex) => (
                        <td
                          key={`${rowIndex}-${colIndex}`}
                          className={`px-3 py-2 ${column === labelColumn ? "bg-blue-50" : ""}`}
                        >
                          {row[column]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <DialogFooter>
            <Button onClick={() => setShowLabelDialog(false)} variant="outline">
              Cancel
            </Button>
            <Button onClick={handleConfirmLabelColumn}>Confirm & Upload</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ImageLoader;
