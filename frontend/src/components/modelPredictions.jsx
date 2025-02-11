import React from "react";

const ModelPredictions = ({ labels, predictions }) => {
  return (
    <div className="bg-gray-50 p-4 rounded-lg">
      <div className="flex justify-between mb-3">
        <span className="font-medium">Model Predictions</span>
        <span className="text-sm text-gray-600">Confidence</span>
      </div>

      {predictions.map(({ label, confidence }) => (
        <div
          key={labels[parseInt(label[label.length - 1])]}
          className="flex justify-between items-center mb-2"
        >
          <span>{labels[parseInt(label[label.length - 1])]}</span>
          <div className="flex items-center gap-2">
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className="text-sm text-gray-600 w-12">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ModelPredictions;
