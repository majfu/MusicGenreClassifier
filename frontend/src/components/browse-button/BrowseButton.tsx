import { BrowseBtn } from "../icons";
import "../../App.css";
import { useRef } from "react";

interface BrowseButtonProps {
  onFileSelect: (file: File) => void;
}

function BrowseButton({ onFileSelect }: BrowseButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

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
