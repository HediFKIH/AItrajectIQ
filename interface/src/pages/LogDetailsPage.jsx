// src/pages/LogDetailsPage.jsx
import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getLog, getProject } from "../api"; // You’ll need to define this in your API
import Accordion from "react-bootstrap/Accordion";

const LogDetailsPage = () => {
  const { projectId, logId } = useParams();
  const [log, setLog] = useState(null);
  const [project, setProject] = useState(null);
  const [error, setError] = useState(null);
  const [error2, setError2] = useState(null);

  useEffect(() => {
    const fetchLog = async () => {
      try {
        const data = await getLog(logId); // Make sure this exists
        setLog(data);
      } catch (err) {
        setError("Failed to fetch log.");
      }
    };

    fetchLog();
  }, [projectId, logId]);

  useEffect(() => {
    const fetchProject = async () => {
      try {
        const data = await getProject(projectId); // Make sure this exists
        setProject(data);
      } catch (err) {
        setError2("Failed to fetch project.");
      }
    };

    fetchProject();
  }, [projectId]);

  if (error) return <div>{error}</div>;
  if (error2) return <div>{error2}</div>;
  if (!log || !project) return <div>Loading...</div>;

  return (
    <div className="m-5">
      <h2 className="mb-4">{project.name}</h2>
      <p className="mb-2">
        <strong>Created:</strong> {new Date(log.timestamp).toLocaleString()}
      </p>
      <p className="mb-5">
        <strong>Log:</strong> {log._id}
      </p>


      <Accordion defaultActiveKey="0">
        {Object.entries(log)
          .filter(([key, value]) => key !== "_id" && Boolean(value)) // filters out falsy values
          .map(([key, value], index) => (
            <Accordion.Item eventKey={index.toString()} key={key}>
              <Accordion.Header>
                <strong>{key}</strong>{" "}
              </Accordion.Header>
              <Accordion.Body>
                {key === "metrics" && Array.isArray(value) ? (
                  <div>
                    {value.map((metric, metricIndex) => (
                      <div key={metricIndex}>
                        {metric.name}: {metric.value}
                      </div>
                    ))}
                  </div>
                ) : (
                  <pre
                    style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
                  >
                    {typeof value === "object"
                      ? JSON.stringify(value, null, 2)
                      : value.toString()}
                  </pre>
                )}
              </Accordion.Body>
            </Accordion.Item>
          ))}
      </Accordion>
    </div>
  );
};

export default LogDetailsPage;
