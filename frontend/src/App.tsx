import {
  Header,
  InstructionIcon,
  IconLeft,
  IconRight,
} from "./components/icons";
import BrowseButton from "./components/browse-button/BrowseButton";
import GoButton from "./components/go-button/GoButton";
import FileNameDisplay from "./components/file-name-display/FileNameDisplay";
import ResultsList from "./components/results-list/ResultsList";
import SvgSpinnerIcon from "./components/icons/SpinnerIcon";
import "./AppLayout.css";
import "./App.css";
import { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [resultsList, setResultsList] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResultsList([]);
  };

  const handleGoClick = () => {
    setIsLoading(true);
    setTimeout(() => {
      const simulatedResults = ["Pop", "Rap"];
      setResultsList(simulatedResults);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className="container">
      {isLoading && (
        <div className="loading-overlay">
          <SvgSpinnerIcon className="custom-spinner" />
        </div>
      )}
      <div className="header">
        <Header />
      </div>
      <div className="instructions">
        <InstructionIcon />
      </div>

      <div className="icons-row">
        <IconLeft />
        <BrowseButton onFileSelect={handleFileSelect} />
        <IconRight />
      </div>

      {selectedFile && (
        <div className="file-selected-display">
          <FileNameDisplay>{selectedFile.name}</FileNameDisplay>
          <GoButton file={selectedFile} onGoClick={handleGoClick} />
        </div>
      )}

      {resultsList.length > 0 && (
        <div className="results">
          <ResultsList resultsList={resultsList} />
        </div>
      )}
    </div>
  );
}

export default App;
