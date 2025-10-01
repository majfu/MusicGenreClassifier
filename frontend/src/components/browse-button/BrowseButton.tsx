import { BrowseBtn } from "../icons";
import "../../App.css";
import { useRef } from "react";

interface BrowseButtonProps {
  onFileSelect: (file: File) => void;
  inputRef?: React.RefObject<HTMLInputElement | null>;
}

function BrowseButton({ onFileSelect, inputRef }: BrowseButtonProps) {
  const internalRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = inputRef || internalRef;

  const openFileBrowser = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div>
      <BrowseBtn className="app-button" onClick={openFileBrowser} />
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".wav,.mp3"
        style={{ display: "none" }}
      />
    </div>
  );
}

export default BrowseButton;
