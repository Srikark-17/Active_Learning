import { Button } from "./ui/button";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

function CheckpointControls({ onSave, onLoad, checkpoints }) {
  return (
    <div className="space-y-4">
      <Button onClick={onSave}>Save Checkpoint</Button>

      <div>
        <Label>Export Checkpoint</Label>
        <Select disabled={checkpoints.length == 0} onValueChange={onLoad}>
          <SelectTrigger>
            <SelectValue placeholder="Select checkpoint" />
          </SelectTrigger>
          <SelectContent>
            {checkpoints.map((cp) => (
              <SelectItem key={cp} value={cp}>
                {cp}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
export default CheckpointControls;
