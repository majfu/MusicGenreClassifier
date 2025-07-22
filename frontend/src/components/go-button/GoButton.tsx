import "../../App.css";
import { GoBtn } from "../icons";

interface GoButtonProps {
  file: File;
  onGoClick: () => void;
}

function GoButton({ file, onGoClick }: GoButtonProps) {
  return <GoBtn className="app-button" onClick={onGoClick} />;
}

export default GoButton;
