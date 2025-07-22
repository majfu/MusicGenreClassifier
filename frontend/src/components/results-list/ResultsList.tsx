import "../../App.css";

interface ResultsListProps {
  resultsList: string[];
}

function ResultsList({ resultsList }: ResultsListProps) {
  if (resultsList.length === 0) {
    return (
      <h1>OOOPS! Looks like we could not classify your audio genre T-T</h1>
    );
  }

  return (
    <>
      <span className="app-text">Your audio was classified as:</span>
      <ul className="list-group">
        {resultsList.map((result) => (
          <li className="list-group-item" key={result}>
            {result}
          </li>
        ))}
      </ul>
    </>
  );
}

export default ResultsList;
