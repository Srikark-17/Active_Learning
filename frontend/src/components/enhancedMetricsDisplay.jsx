import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
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

const EnhancedMetricsDisplay = ({ metrics, episode_history, lr_history }) => {
  // For debugging
  console.log("Metrics received:", metrics);

  // Transform loss data directly from the API format
  const lossData =
    metrics?.currentEpochLosses?.x?.map((x, i) => ({
      epoch: x,
      loss: metrics.currentEpochLosses.y[i],
    })) || [];

  // Transform accuracy data
  const accuracyData =
    metrics?.episodeAccuracies?.x?.map((x, i) => ({
      episode: x,
      accuracy: metrics.episodeAccuracies.y[i],
    })) || [];

  console.log("Transformed loss data:", lossData);

  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle>Training Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-8">
        {/* Current Episode and Best Accuracy */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <h3 className="font-medium text-sm text-gray-600">
              Current Episode
            </h3>
            <p className="text-2xl font-bold">
              {metrics?.current_episode || 0}
            </p>
          </div>
          <div>
            <h3 className="font-medium text-sm text-gray-600">
              Best Validation Accuracy
            </h3>
            <p className="text-2xl font-bold">
              {metrics?.best_val_acc?.toFixed(2) || 0}%
            </p>
          </div>
        </div>

        {/* Training Loss Chart */}
        {lossData.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Training Loss</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lossData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  label={{
                    value: "Loss",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#82ca9d"
                  name="Training Loss"
                  dot={false} // Remove dots for cleaner look with many points
                  activeDot={{ r: 8 }} // Larger dot on hover
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Episode Accuracy Chart - only show if there's data */}
        {accuracyData.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Episode Accuracy</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={accuracyData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="episode"
                  label={{ value: "Episode", position: "bottom" }}
                />
                <YAxis
                  domain={[0, 100]}
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#8884d8"
                  name="Validation Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Learning Rate Chart - only show if there's data */}
        {lr_history && lr_history.length > 0 && (
          <div className="h-80">
            <h3 className="font-medium mb-4">Learning Rate</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={lr_history}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: "Epoch", position: "bottom" }}
                />
                <YAxis
                  scale="log"
                  label={{
                    value: "Learning Rate",
                    angle: -90,
                    position: "insideLeft",
                    style: { textAnchor: "middle" },
                  }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="new_lr"
                  stroke="#8884d8"
                  name="Learning Rate"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Episode History */}
        {episode_history?.length > 0 && (
          <div className="mt-4">
            <h3 className="font-medium mb-2">Episode History</h3>
            <div className="space-y-2">
              {episode_history.map((episode, idx) => (
                <div key={idx} className="text-sm">
                  <div className="flex justify-between">
                    <span>Episode {episode.episode}</span>
                    <span>
                      Best Acc:{" "}
                      {episode.best_val_acc
                        ? episode.best_val_acc.toFixed(2)
                        : episode.train_result?.best_accuracy?.toFixed(2) ||
                          "0.00"}
                      %
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default EnhancedMetricsDisplay;
