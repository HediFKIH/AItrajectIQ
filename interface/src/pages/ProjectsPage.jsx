// src/pages/ProjectsPage.jsx
import React from "react";
import ProjectList from "../components/ProjectList";

const ProjectsPage = () => {
  return (
    <div className="m-5">
      <h2 className="mb-4">Your Projects</h2>
      <ProjectList userId={"68b8ca4d993be141377a5018"} />
    </div>
  );
};

export default ProjectsPage;
