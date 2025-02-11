// components/MetricsDisplay.jsx
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";

const MetricsDisplay = ({ metrics }) => {
  const { episodeAccuracies, currentEpochLosses } = metrics;

  if (
    !metrics.episodeAccuracies?.length &&
    !metrics.currentEpochLosses?.length
  ) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Accuracy Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Active Learning Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={episodeAccuracies}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="x"
                  label={{ value: "Episode", position: "bottom" }}
                />
                <YAxis
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#8884d8"
                  name="Validation Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Loss Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={currentEpochLosses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="x"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#82ca9d"
                  name="Training Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MetricsDisplay;
