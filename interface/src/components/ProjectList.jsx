// src/components/ProjectList.jsx
import React, { useEffect, useState } from "react";
import { getProjects } from "../api";
import ListGroup from "react-bootstrap/ListGroup";
import { useNavigate } from "react-router-dom";

const ProjectList = ({ userId }) => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const data = await getProjects(userId);
        setProjects(data);
      } catch (err) {
        setError("Failed to load projects.");
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, [userId]);

  const handleSelect = (project) => {
    navigate(`/projects/${project._id}`);
  };

  if (loading) return <p>Loading projects...</p>;
  if (error) return <Alert variant="danger">{error}</Alert>;

  return (
    <ListGroup className="w-50">
      {projects.map((project) => (
        <ListGroup.Item
          key={project._id}
          action
          onClick={() => handleSelect(project)}
        >
          {project.name}
        </ListGroup.Item>
      ))}
    </ListGroup>
  );
};

export default ProjectList;
