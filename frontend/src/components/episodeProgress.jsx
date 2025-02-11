import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

const EpisodeProgress = ({ episode, stats }) => {
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="text-lg">Episode {episode} Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Initial Labeled Size:</span>
            <span>{stats.initial_labeled_size}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Current Labeled Size:</span>
            <span>{stats.current_labeled_size}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Best Validation Accuracy:</span>
            <span>{(stats.best_val_acc * 100).toFixed(2)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{
                width: `${
                  (stats.current_labeled_size / stats.target_size) * 100
                }%`,
              }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default EpisodeProgress;
