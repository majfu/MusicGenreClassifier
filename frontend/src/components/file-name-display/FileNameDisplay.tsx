import SvgSelectedFileDisplay from "../icons/SelectedFileDisplayIcon";
import "./FileNameDisplay.css";

interface SelectedFileDisplayProps {
  children: React.ReactNode;
}

function FileNameDisplay({ children }: SelectedFileDisplayProps) {
  return (
    <div className="selected-file-wrapper">
      <SvgSelectedFileDisplay className="selected-file-icon" />
      <span className="selected-file-name">{children}</span>
    </div>
  );
}

export default FileNameDisplay;
