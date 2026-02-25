import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
});

export const getProjects = async (userId) => {
  const params = userId ? { user_id: userId } : {};
  const response = await api.get("/projects/", { params });
  return response.data;
};

// Get logs for a project
export const getLogs = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  const response = await api.get("/logs/", { params });
  return response.data;
};

// Get a specific log
export const getLog = async (logId) => {
  const response = await api.get(`/logs/${logId}`);
  return response.data;
};

export const getProject = async (projectId) => {
  const response = await api.get(`/projects/${projectId}`);
  return response.data;
};

export const getRecords = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  const response = await api.get("/records/", { params });
  return response.data;
};

export default api;
