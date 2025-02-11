// components/ImageLoader.jsx
import React, { useRef } from "react";
import { Button } from "./ui/button";
import { Upload } from "lucide-react";

const ImageLoader = ({ onImagesLoaded, onError }) => {
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    processFiles(files);
  };

  const handleFolderChange = (event) => {
    const files = Array.from(event.target.files);
    processFiles(files);
  };

  const processFiles = (files) => {
    // Validate files are images
    const validFiles = files.filter(
      (file) =>
        file.type.startsWith("image/") ||
        file.name.match(/\.(jpg|jpeg|png|gif|bmp|webp)$/i)
    );

    if (validFiles.length === 0) {
      onError("Please select valid image files");
      return;
    }

    if (validFiles.length !== files.length) {
      onError("Some files were not images and were removed");
    }

    onImagesLoaded(validFiles);
  };

  return (
    <div className="space-y-4">
      {/* Hidden file inputs */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        multiple
        accept="image/*"
        className="hidden"
      />
      <input
        type="file"
        ref={folderInputRef}
        onChange={handleFolderChange}
        // This enables folder selection
        {...{ webkitdirectory: "", directory: "" }}
        className="hidden"
      />

      {/* Upload buttons */}
      <div className="flex gap-2">
        <Button onClick={() => fileInputRef.current.click()} className="flex-1">
          <Upload className="h-4 w-4 mr-2" />
          Upload Images
        </Button>
        <Button
          onClick={() => folderInputRef.current.click()}
          className="flex-1"
        >
          <Upload className="h-4 w-4 mr-2" />
          Upload Folder
        </Button>
      </div>
    </div>
  );
};

export default ImageLoader;
