import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
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
        <h2 className="mb-4">
          Explainability Logs for <b>{project.name}</b>
        </h2>
        <h5 className="mb-4">Model: {project.model_name}</h5>
        <RecordList projectId={projectId} />
      </div>
    );
  } else {
    return (
      <div className="m-5">
        <h2 className="mb-4">
          Observability Logs for <b>{project.name}</b>
        </h2>
        <LogList projectId={projectId} />
      </div>
    );
  }
};

export default LogsPage;
