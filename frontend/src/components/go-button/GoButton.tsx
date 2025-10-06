import "../../App.css";
import { GoBtn } from "../icons";

interface GoButtonProps {
  onGoClick: () => void;
}

function GoButton({ onGoClick }: GoButtonProps) {
  return <GoBtn className="app-button" onClick={onGoClick} />;
}

export default GoButton;
