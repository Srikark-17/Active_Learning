import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "../components/ui/button";

const TrainingControls = ({
  onStartTraining,
  onStartNewBatch,
  isRetraining,
  batchCompleted,
  trainingMetrics,
}) => {
  return (
    <Card className="mb-4">
      <CardHeader>
        <CardTitle>Training Controls</CardTitle>
      </CardHeader>
      <CardContent>
        {isRetraining ? (
          <div className="space-y-2">
            <div className="animate-pulse flex space-x-4 items-center">
              <div className="rounded-full bg-blue-400 h-3 w-3"></div>
              <div className="flex-1">
                <p className="text-blue-600">Training in progress...</p>
                {trainingMetrics && (
                  <p className="text-sm text-gray-600">
                    Current Epoch: {trainingMetrics.current_epoch}/
                    {trainingMetrics.total_epochs}
                  </p>
                )}
              </div>
            </div>
          </div>
        ) : batchCompleted ? (
          <div className="space-y-4">
            <Button onClick={onStartTraining} className="w-full">
              Start Training
            </Button>
          </div>
        ) : (
          <Button onClick={onStartNewBatch} className="w-full">
            Start New Batch
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

export default TrainingControls;
