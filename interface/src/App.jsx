// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ProjectsPage from "./pages/ProjectsPage";
import LogsPage from "./pages/LogsPage";
import LogDetailsPage from "./pages/LogDetailsPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<LogsPage />} />
        <Route
          path="/projects/:projectId/:logId"
          element={<LogDetailsPage />}
        />
      </Routes>
    </Router>
  );
}

export default App;
