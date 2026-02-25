import React, { useEffect, useState } from "react";
import { ListGroup, Spinner, Alert } from "react-bootstrap";
import { getLogs } from "../api"; // You need to define this
import { useNavigate } from "react-router-dom";

const LogList = ({ projectId }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const data = await getLogs(projectId);
        console.log("Fetched logs:", data);
        setLogs(data);
      } catch (err) {
        setError("Failed to load logs.");
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [projectId]);

  const handleSelect = (log) => {
    navigate(`/projects/${projectId}/${log._id}`);
  };

  if (loading) return <p>Loading logs...</p>;
  if (error) return <Alert variant="danger">{error}</Alert>;

  return (
    <ListGroup>
      {logs.map((log) => (
        <ListGroup.Item key={log._id} action onClick={() => handleSelect(log)}>
          {log.prompt}
        </ListGroup.Item>
      ))}
    </ListGroup>
  );
};

export default LogList;
