import React from "react";
import { Alert, AlertDescription } from "./ui/alert";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Label } from "./ui/label";
import { Progress } from "@/components/ui/progress";
import { Loader2 } from "lucide-react";
import EnhancedMetricsDisplay from "./enhancedMetricsDisplay";

const AutomatedTrainingControls = ({
  onStart,
  onStop,
  metrics,
  status,
  disabled,
  episode_history,
}) => {
  // Helper function to determine training phase
  const getTrainingPhase = () => {
    if (!status?.is_training) return null;
    if (status?.current_batch?.labeled < status?.current_batch?.total) {
      return "labeling";
    }
    return "training";
  };

  const phase = getTrainingPhase();

  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          Active Learning Progress
          <div className="flex gap-2">
            <Button
              onClick={onStart}
              disabled={disabled || status?.is_training}
              variant={status?.is_training ? "secondary" : "default"}
            >
              {status?.is_training ? "Training Active" : "Start Training"}
            </Button>
            <Button
              onClick={onStop}
              disabled={disabled || !status?.is_training}
              variant="outline"
            >
              Stop Training
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {status && (
          <div className="space-y-4">
            {/* Training Phase Indicator */}
            {phase && (
              <Alert variant={phase === "training" ? "default" : "secondary"}>
                <div className="flex items-center gap-2">
                  {phase === "training" && (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  )}
                  <AlertDescription>
                    {phase === "labeling"
                      ? "Label all images in the current batch to continue training"
                      : "Training in progress - please wait..."}
                  </AlertDescription>
                </div>
              </Alert>
            )}

            {/* Current Episode Info */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Current Episode</Label>
                <div className="text-2xl font-semibold">
                  {status.current_episode}
                </div>
              </div>
            </div>

            {/* Dataset Statistics */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Labeled Images</Label>
                <div className="text-lg">{status.labeled_count}</div>
              </div>
              <div>
                <Label>Remaining Unlabeled</Label>
                <div className="text-lg">{status.unlabeled_count}</div>
              </div>
            </div>

            {/* Training Metrics */}
            {metrics && (
              <div className="mt-6 space-y-4 pt-4 border-t">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Validation Accuracy</Label>
                    <div className="text-2xl font-semibold">
                      {(metrics.best_val_acc || 0).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <Label>Total Episodes</Label>
                    <div className="text-2xl font-semibold">
                      {metrics.current_episode || 0}
                    </div>
                  </div>
                </div>

                {/* Training Progress Visualization */}
                {phase === "training" &&
                  metrics.current_epoch_losses?.x?.length > 0 && (
                    <div>
                      <Label>Training Progress</Label>
                      <Progress
                        value={
                          (metrics.current_epoch_losses.x.length /
                            metrics.total_epochs) *
                          100
                        }
                        className="my-2"
                      />
                      <div className="text-sm text-gray-500">
                        Epoch {metrics.current_epoch_losses.x.length} of{" "}
                        {metrics.total_epochs}
                      </div>
                    </div>
                  )}

                {/* Metrics Display */}
                <EnhancedMetricsDisplay
                  metrics={{
                    episodeAccuracies: metrics.episode_accuracies || {
                      x: [],
                      y: [],
                    },
                    currentEpochLosses: metrics.current_epoch_losses || {
                      x: [],
                      y: [],
                    },
                  }}
                  episode_history={episode_history}
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AutomatedTrainingControls;
