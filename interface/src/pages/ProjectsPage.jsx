// src/pages/ProjectsPage.jsx
import React from "react";
import ProjectList from "../components/ProjectList";

const ProjectsPage = () => {
  return (
    <div className="m-5">
      <h2 className="mb-4">Your Projects</h2>
      <ProjectList userId={"6889cc16d221921197a18c48"} />
    </div>
  );
};

export default ProjectsPage;
