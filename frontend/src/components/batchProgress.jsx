import React from "react";

const BatchProgress = ({ stats }) => {
  const progressPercentage = (stats.completed / stats.totalImages) * 100;

  return (
    <div className="bg-gray-50 p-4 rounded-lg">
      <div className="flex justify-between mb-2">
        <span className="font-medium">Current Batch Progress</span>
        <span className="text-sm text-blue-600">
          {stats.completed}/{stats.totalImages}
        </span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
        <div
          className="bg-blue-600 h-2 rounded-full"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>Images Remaining: {stats.remaining}</div>
        <div>Time Elapsed: {stats.timeElapsed}</div>
      </div>
    </div>
  );
};

export default BatchProgress;
