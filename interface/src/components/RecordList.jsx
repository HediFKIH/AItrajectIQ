import React, { useEffect, useState } from "react";
import { ListGroup, Spinner, Alert } from "react-bootstrap";
import { getRecords } from "../api"; // You need to define this
import { useNavigate } from "react-router-dom";

function getTokenStyle(relevance) {
  const opacity = Math.abs(relevance).toFixed(2); // range 0 to 1

  if (relevance > 0) {
    return {
      backgroundColor: `rgba(0, 255, 0, ${opacity})`, // green
      borderRadius: "4px",
      padding: "2px 4px",
      margin: "0 2px",
    };
  } else if (relevance < 0) {
    return {
      backgroundColor: `rgba(255, 0, 0, ${opacity})`, // red
      borderRadius: "4px",
      padding: "2px 4px",
      margin: "0 2px",
    };
  } else {
    return {
      margin: "0 2px",
    };
  }
}

const RecordList = ({ projectId }) => {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchRecords = async () => {
      try {
        const data = await getRecords(projectId);
        console.log("Fetched records:", data);
        setRecords(data);
      } catch (err) {
        setError("Failed to load records.");
      } finally {
        setLoading(false);
      }
    };

    fetchRecords();
  }, [projectId]);

  if (loading) return <p>Loading records...</p>;
  if (error) return <Alert variant="danger">{error}</Alert>;

  return (
    <ListGroup>
      {records.map((record) => (
        <ListGroup.Item key={record._id} action>
          <div style={{ display: "flex", flexWrap: "wrap" }}>
            {record.tokens.slice(1, -1).map((token, i) => (
              <span key={i} style={getTokenStyle(record.relevances[i + 1])}>
                {token}
              </span>
            ))}
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: "8px",
              fontStyle: "italic",
            }}
          >
            <span>output: {record.Output}</span>
          </div>
        </ListGroup.Item>
      ))}
    </ListGroup>
  );
};

export default RecordList;
