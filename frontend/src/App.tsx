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
import { useEffect, useRef, useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [resultsList, setResultsList] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const resultsEndRef = useRef<HTMLDivElement>(null);
  const selectedEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResultsList([]);
    setErrorMessage(null);
  };

  const clearData = () => {
    setSelectedFile(null);
    setResultsList([]);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleGoClick = async () => {
    if (!selectedFile) return;

    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/prediction", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        setErrorMessage(data.message);
      } else {
        const data = await response.json();
        setResultsList(data.genres);
      }
    } catch (error) {
      setErrorMessage("Error uploading file");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    resultsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    selectedEndRef.current?.scrollIntoView({ behavior: "smooth" });
  });

  return (
    <div className="container">
      {isLoading && (
        <div className="loading-overlay">
          <SvgSpinnerIcon className="custom-spinner" />
        </div>
      )}

      {errorMessage && (
        <div className="loading-overlay">
          <div className="error-message">
            {errorMessage}
            <button
              className="error-close-btn"
              onClick={() => {
                clearData();
              }}
            >
              X
            </button>
          </div>
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
        <BrowseButton onFileSelect={handleFileSelect} inputRef={fileInputRef} />
        <IconRight />
      </div>

      {selectedFile && (
        <div className="file-selected-display">
          <FileNameDisplay>{selectedFile.name}</FileNameDisplay>
          <GoButton file={selectedFile} onGoClick={handleGoClick} />
          <div ref={selectedEndRef} />
        </div>
      )}

      {resultsList.length > 0 && (
        <div className="results">
          <ResultsList resultsList={resultsList} />
          <button
            className="reset-btn"
            onClick={() => {
              clearData();
            }}
          >
            Reset
          </button>
          <div ref={resultsEndRef} />
        </div>
      )}
    </div>
  );
}

export default App;
