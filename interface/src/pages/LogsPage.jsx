import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Button } from "react-bootstrap";
import LogList from "../components/LogList";
import RecordList from "../components/RecordList";
import { getProject } from "../api";

const LogsPage = () => {
  const { projectId } = useParams();
  const [project, setProject] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProject = async () => {
      try {
        const data = await getProject(projectId);
        setProject(data);
      } catch (err) {
        setError("Failed to fetch project.");
      }
    };

    fetchProject();
  }, [projectId]);

  if (error) return <Alert variant="danger">{error}</Alert>;
  if (!project) return <p>...</p>;

  if (project.type === "LLMExplain") {
    return (
      <div className="m-5">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <div>
            <h2 className="mb-2">
              Logs for <b>{project.name}</b>
            </h2>
            <h5 className="mb-0">Model: {project.model_name}</h5>
          </div>
          <Link to={`/dashboard/${projectId}`}>
            <Button variant="primary">View Dashboard</Button>
          </Link>
        </div>
        <RecordList projectId={projectId} />
      </div>
    );
  } else {
    return (
      <div className="m-5">
        <div className="d-flex justify-content-between align-items-center mb-4">
          <h2>
            Logs for <b>{project.name}</b>
          </h2>
          <Link to={`/dashboard/${projectId}`}>
            <Button variant="primary">View Dashboard</Button>
          </Link>
        </div>
        <LogList projectId={projectId} />
      </div>
    );
  }
};

export default LogsPage;