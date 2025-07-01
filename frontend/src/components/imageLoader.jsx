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
  DialogTrigger,
} from "./ui/dialog";
import Papa from "papaparse";

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

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files).filter((file) =>
      file.type.startsWith("image/")
    );

    if (files.length > 0) {
      onImagesLoaded(files);
    } else {
      onError("No valid image files selected");
    }
  };

  const handleFolderChange = (e) => {
    const files = Array.from(e.target.files).filter((file) =>
      file.type.startsWith("image/")
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

    // Separate CSV and image files
    const csvFiles = files.filter(
      (file) =>
        file.name.endsWith(".csv") ||
        file.name.endsWith(".tsv") ||
        file.name.endsWith(".txt")
    );

    const imgFiles = files.filter((file) => file.type.startsWith("image/"));

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

    try {
      console.log(
        `Processing ${csvFiles.length} CSV files and ${imgFiles.length} image files`
      );

      // For direct upload, we just pass all the files together
      const allFiles = [...csvFiles, ...imgFiles];

      // We'll use a special flag to indicate this is a combined upload
      onImagesLoaded(allFiles, "combined");

      setIsLoading(false);
    } catch (error) {
      onError(`Error processing files: ${error.message}`);
      setIsLoading(false);
    }
  };

  // In ImageLoader.jsx, modify handleCsvWithLabelsSelect
  const handleCsvWithLabelsSelect = async (e) => {
    setIsLoading(true);
    const files = Array.from(e.target.files);

    if (files.length === 0) {
      setIsLoading(false);
      return;
    }

    // Separate CSV and image files
    const csvFiles = files.filter(
      (file) =>
        file.name.endsWith(".csv") ||
        file.name.endsWith(".tsv") ||
        file.name.endsWith(".txt")
    );

    const imgFiles = files.filter((file) => file.type.startsWith("image/"));

    if (csvFiles.length === 0) {
      onError("Please include a CSV file with your selection");
      setIsLoading(false);
      return;
    }

    try {
      // Parse the CSV to detect column names
      const csvFile = csvFiles[0];
      setCsvFile(csvFile);

      if (imgFiles.length > 0) {
        setImageFiles(imgFiles);
      } else {
        // If no images selected, make this a CSV-only operation
        setImageFiles([]);
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          // Parse just the first few rows to get column names and preview
          Papa.parse(content, {
            header: true,
            preview: 5,
            complete: (result) => {
              if (result.errors.length > 0) {
                onError(
                  `Error parsing CSV preview: ${result.errors[0].message}`
                );
                setIsLoading(false);
                return;
              }

              // Auto-detect label column
              const columns = result.meta.fields;

              // Look for common label column names, with 'annotation' added to the list
              const labelColumns = [
                "label",
                "class",
                "category",
                "target",
                "classification",
                "annotation",
              ];
              let detectedLabelColumn = null;

              for (const col of labelColumns) {
                if (columns.includes(col)) {
                  detectedLabelColumn = col;
                  break;
                }
                // Try case-insensitive match
                const colMatch = columns.find((c) => c.toLowerCase() === col);
                if (colMatch) {
                  detectedLabelColumn = colMatch;
                  break;
                }
              }

              if (detectedLabelColumn) {
                setLabelColumn(detectedLabelColumn);
              } else if (columns.length > 1) {
                // If we couldn't find a standard label column, default to the second column
                setLabelColumn(columns[1]);
              }

              // Extract unique label values for auto-populating labels
              const uniqueLabels = new Set();
              for (const row of result.data) {
                if (detectedLabelColumn && row[detectedLabelColumn]) {
                  uniqueLabels.add(row[detectedLabelColumn].trim());
                }
              }

              // Store the unique labels
              setDetectedLabels(
                Array.from(uniqueLabels).filter((label) => label)
              );

              // Store preview for display
              setCsvPreview(result.data);

              // Open dialog to confirm label column
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

      reader.readAsText(csvFile);
    } catch (error) {
      onError(`Error processing files: ${error.message}`);
      setIsLoading(false);
    }
  };

  // Add state for detected labels
  const [detectedLabels, setDetectedLabels] = useState([]);

  // In handleConfirmLabelColumn, add detected labels to the response
  const handleConfirmLabelColumn = () => {
    // Check if we have image files or CSV only
    const csvOnly = imageFiles.length === 0;

    if (csvOnly) {
      // For CSV-only mode, we'll use a special flag
      onImagesLoaded([csvFile], "csv-with-labels", labelColumn, detectedLabels);
    } else {
      // For combined mode with images included
      const allFiles = [csvFile, ...imageFiles];
      onImagesLoaded(
        allFiles,
        "combined-with-labels",
        labelColumn,
        detectedLabels
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
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Images
          </Button>
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
          >
            <FolderUp className="h-4 w-4 mr-2" />
            Upload Folder
          </Button>
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
          >
            <Archive className="h-4 w-4 mr-2" />
            {isLoading ? "Processing..." : "Upload CSV + Images Together"}
          </Button>
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
          >
            <Table className="h-4 w-4 mr-2" />
            {isLoading ? "Processing..." : "Upload CSV with Labels + Images"}
          </Button>
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

      {/* Dialog for confirming label column */}
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
          </div>

          {csvPreview && (
            <div className="max-h-60 overflow-y-auto">
              <h4 className="font-medium mb-2">CSV Preview</h4>
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(csvPreview[0]).map((column) => (
                      <th
                        key={column}
                        className={`px-3 py-2 text-left font-medium text-gray-500 ${
                          column === labelColumn ? "bg-blue-100" : ""
                        }`}
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
                          className={`px-3 py-2 ${
                            column === labelColumn ? "bg-blue-50" : ""
                          }`}
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
