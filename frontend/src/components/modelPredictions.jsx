import React from "react";

const ModelPredictions = ({ predictions = [], labels = [] }) => {
  // Debug logging
  console.log("ModelPredictions Debug:");
  console.log("predictions:", predictions);
  console.log("labels:", labels);
  console.log("labels.length:", labels.length);

  // If no labels are defined by user, don't show predictions
  if (!labels || labels.length === 0) {
    return <div className="text-sm text-gray-500">No labels defined yet</div>;
  }

  // Only show predictions for the user-defined labels (first N classes)
  const userDefinedPredictions = predictions.slice(0, labels.length);

  console.log("userDefinedPredictions:", userDefinedPredictions);

  // Filter out predictions with very low confidence and only show user-defined classes
  const filteredPredictions = userDefinedPredictions
    .map((pred, idx) => {
      const mappedPred = {
        ...pred,
        labelText: labels[idx] || `Class ${idx}`,
        labelIndex: idx,
      };
      console.log(`Mapping prediction ${idx}:`, mappedPred);
      return mappedPred;
    })
    .filter((pred) => {
      const confidence = Math.round(pred.confidence * 100);
      const shouldShow = confidence > 0 && !pred.labelText.startsWith("Class");
      console.log(
        `Filter check for ${pred.labelText}: confidence=${confidence}%, shouldShow=${shouldShow}`
      );
      return shouldShow;
    });

  console.log("filteredPredictions:", filteredPredictions);

  if (!filteredPredictions || filteredPredictions.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        No meaningful predictions available
      </div>
    );
  }

  // Sort by confidence (highest first)
  const sortedPredictions = filteredPredictions.sort(
    (a, b) => b.confidence - a.confidence
  );

  return (
    <div>
      <h4 className="font-medium mb-2">Model Predictions</h4>
      <div className="space-y-2">
        {sortedPredictions.map((pred, idx) => {
          const confidence = Math.round(pred.confidence * 100);

          console.log(
            `Rendering prediction ${idx}:`,
            pred.labelText,
            confidence
          );

          return (
            <div key={idx} className="flex items-center gap-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${confidence}%` }}
                ></div>
              </div>
              <span className="text-sm whitespace-nowrap">
                {pred.labelText}: {confidence}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelPredictions;
