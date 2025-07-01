import React from "react";

// In ModelPredictions component
const ModelPredictions = ({ predictions = [], labels = [] }) => {
  // Filter out predictions with 0% probability
  const filteredPredictions = predictions.filter(
    (pred) => Math.round(pred.confidence * 100) > 0
  );

  if (!filteredPredictions || filteredPredictions.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        No meaningful predictions available
      </div>
    );
  }

  return (
    <div>
      <h4 className="font-medium mb-2">Model Predictions</h4>
      <div className="space-y-2">
        {filteredPredictions.map((pred, idx) => {
          const labelIndex = parseInt(pred.label.replace("Label ", ""));
          const labelText = labels[labelIndex] || pred.label;
          const confidence = Math.round(pred.confidence * 100);

          // Skip if confidence is 0%
          if (confidence === 0) return null;

          return (
            <div key={idx} className="flex items-center gap-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${confidence}%` }}
                ></div>
              </div>
              <span className="text-sm whitespace-nowrap">
                {labelText}: {confidence}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelPredictions;
