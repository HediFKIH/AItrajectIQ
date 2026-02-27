import React, { useEffect, useState, useMemo } from "react";
import { useParams, Link } from "react-router-dom";
import { Button, Card, Row, Col, Form, Alert } from "react-bootstrap";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getProject, getLogs } from "../api";

const DashboardPage = () => {
  const { projectId } = useParams();
  const [project, setProject] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState("all"); // all, 24h, 7d, 30d
  const [dateRange, setDateRange] = useState({ start: "", end: "" });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [projectData, logsData] = await Promise.all([
          getProject(projectId),
          getLogs(projectId)
        ]);
        setProject(projectData);
        setLogs(logsData);
      } catch (err) {
        setError("Failed to fetch project data.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [projectId]);

  // Filter logs based on time range
  const filteredLogs = useMemo(() => {
    if (!logs || logs.length === 0) return [];
    
    let filtered = [...logs];
    const now = new Date();
    
    if (timeRange !== "all") {
      const hoursMap = { "24h": 24, "7d": 24 * 7, "30d": 24 * 30 };
      const hoursBack = hoursMap[timeRange];
      const cutoff = new Date(now.getTime() - hoursBack * 60 * 60 * 1000);
      filtered = filtered.filter(log => new Date(log.timestamp) >= cutoff);
    }
    
    if (dateRange.start && dateRange.end) {
      const start = new Date(dateRange.start);
      const end = new Date(dateRange.end);
      filtered = filtered.filter(log => {
        const logDate = new Date(log.timestamp);
        return logDate >= start && logDate <= end;
      });
    }
    
    return filtered.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [logs, timeRange, dateRange]);

  // Get unique metrics across all logs
  const uniqueMetrics = useMemo(() => {
    const metrics = new Set();
    filteredLogs.forEach(log => {
      if (log.metrics && Array.isArray(log.metrics)) {
        log.metrics.forEach(metric => metrics.add(metric.name));
      }
    });
    return Array.from(metrics);
  }, [filteredLogs]);

  // Get hallucination data
  const hallucinationData = useMemo(() => {
    const hallucinations = [];
    filteredLogs.forEach(log => {
      if (log.metrics && Array.isArray(log.metrics)) {
        const hallucinationMetric = log.metrics.find(m => m.name === "hallucination");
        if (hallucinationMetric && hallucinationMetric.value) {
          hallucinations.push({
            timestamp: log.timestamp,
            value: hallucinationMetric.value
          });
        }
      }
    });
    return hallucinations;
  }, [filteredLogs]);

  // Prepare chart data
  const prepareChartData = (metricName = null) => {
    return filteredLogs.map(log => {
      const timestamp = new Date(log.timestamp).toLocaleString();
      const data = { timestamp, time: new Date(log.timestamp).getTime() };
      
      if (metricName) {
        // For specific metric
        const metric = log.metrics?.find(m => m.name === metricName);
        data.value = metric ? parseFloat(metric.value) || null : null;
      } else {
        // For latency and tokens
        data.latency = log.latency || null;
        data.n_tokens = log.n_tokens || null;
        data.latency_per_token = (log.n_tokens && log.latency) ? log.latency / log.n_tokens : null;
      }
      
      return data;
    }).filter(item => {
      // Only filter out if ALL relevant values are null
      if (metricName) {
        return item.value !== null && item.value !== undefined;
      } else {
        return item.latency !== null || item.n_tokens !== null || item.latency_per_token !== null;
      }
    });
  };

  // Calculate mean values
  const calculateMean = (data, key) => {
    const validValues = data.filter(d => d[key] !== null && d[key] !== undefined);
    if (validValues.length === 0) return 0;
    const sum = validValues.reduce((acc, d) => acc + parseFloat(d[key]), 0);
    return (sum / validValues.length).toFixed(2);
  };

  if (error) return <Alert variant="danger">{error}</Alert>;
  if (loading) return <p>Loading...</p>;
  if (!project) return <p>Project not found</p>;

  const baseChartData = prepareChartData();

  return (
    <div className="m-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h2 className="mb-2">
            Dashboard for <b>{project.name}</b>
          </h2>
          {project.model_name && (
            <h5 className="text-muted">Model: {project.model_name}</h5>
          )}
        </div>
        <Link to={`/logs/${projectId}`}>
          <Button variant="outline-primary">Back to Logs</Button>
        </Link>
      </div>

      {/* Filters */}
      <Card className="mb-4">
        <Card.Header>
          <h5>Filters</h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={4}>
              <Form.Group>
                <Form.Label>Time Range</Form.Label>
                <Form.Select 
                  value={timeRange} 
                  onChange={(e) => setTimeRange(e.target.value)}
                >
                  <option value="all">All Time</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={4}>
              <Form.Group>
                <Form.Label>Start Date</Form.Label>
                <Form.Control
                  type="datetime-local"
                  value={dateRange.start}
                  onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
                />
              </Form.Group>
            </Col>
            <Col md={4}>
              <Form.Group>
                <Form.Label>End Date</Form.Label>
                <Form.Control
                  type="datetime-local"
                  value={dateRange.end}
                  onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
                />
              </Form.Group>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {filteredLogs.length === 0 ? (
        <Alert variant="info">No logs found for the selected time range.</Alert>
      ) : (
        <>
          {/* Core Metrics */}
          <Row>
            <Col md={4}>
              <Card className="mb-4">
                <Card.Header>
                  <h5>Latency Over Time</h5>
                </Card.Header>
                <Card.Body>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={baseChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        type="number" 
                        scale="time" 
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(time) => new Date(time).toLocaleString()}
                        formatter={(value) => [`${value}ms`, 'Latency']}
                      />
                      <Line type="monotone" dataKey="latency" stroke="#8884d8" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="text-center mt-2">
                    <strong>Mean Latency: {calculateMean(baseChartData, 'latency')}ms</strong>
                  </div>
                </Card.Body>
              </Card>
            </Col>

            <Col md={4}>
              <Card className="mb-4">
                <Card.Header>
                  <h5>Tokens Over Time</h5>
                </Card.Header>
                <Card.Body>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={baseChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        type="number" 
                        scale="time" 
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(time) => new Date(time).toLocaleString()}
                        formatter={(value) => [`${value}`, 'Tokens']}
                      />
                      <Line type="monotone" dataKey="n_tokens" stroke="#82ca9d" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="text-center mt-2">
                    <strong>Mean Tokens: {calculateMean(baseChartData, 'n_tokens')}</strong>
                  </div>
                </Card.Body>
              </Card>
            </Col>

            <Col md={4}>
              <Card className="mb-4">
                <Card.Header>
                  <h5>Latency per Token</h5>
                </Card.Header>
                <Card.Body>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={baseChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="time" 
                        type="number" 
                        scale="time" 
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(time) => new Date(time).toLocaleString()}
                        formatter={(value) => [`${value.toFixed(3)}ms`, 'Latency per Token']}
                      />
                      <Line type="monotone" dataKey="latency_per_token" stroke="#ffc658" strokeWidth={2} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div className="text-center mt-2">
                    <strong>Mean Latency per Token: {calculateMean(baseChartData, 'latency_per_token')}ms</strong>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* Dynamic Metric Charts */}
          {uniqueMetrics.length > 0 && (
            <>
              <h3 className="mb-3">Custom Metrics</h3>
              
              {/* Hallucination List */}
              {uniqueMetrics.includes("hallucination") && hallucinationData.length > 0 && (
                <Row>
                  <Col md={12}>
                    <Card className="mb-4">
                      <Card.Header>
                        <h5>Hallucinations Detected</h5>
                      </Card.Header>
                      <Card.Body>
                        <div style={{ maxHeight: "300px", overflowY: "auto" }}>
                          {hallucinationData.map((item, index) => (
                            <div key={index} className="border-bottom py-2">
                              <div className="d-flex justify-content-between">
                                <span className="text-muted">
                                  {new Date(item.timestamp).toLocaleString()}
                                </span>
                              </div>
                              <div className="mt-1">
                                <strong>Hallucination:</strong> {item.value}
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="text-center mt-3">
                          <strong>Total Hallucinations: {hallucinationData.length}</strong>
                        </div>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              )}

              <Row>
                {uniqueMetrics.filter(metric => metric !== "hallucination").map((metricName, index) => {
                  const metricData = prepareChartData(metricName);
                  return (
                    <Col md={6} key={metricName}>
                      <Card className="mb-4">
                        <Card.Header>
                          <h5>{metricName} Over Time</h5>
                        </Card.Header>
                        <Card.Body>
                          <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={metricData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                dataKey="time" 
                                type="number" 
                                scale="time" 
                                domain={['dataMin', 'dataMax']}
                                tickFormatter={(time) => new Date(time).toLocaleTimeString()}
                              />
                              <YAxis />
                              <Tooltip 
                                labelFormatter={(time) => new Date(time).toLocaleString()}
                                formatter={(value) => [value, metricName]}
                              />
                              <Line 
                                type="monotone" 
                                dataKey="value" 
                                stroke={`hsl(${index * 137.5 % 360}, 70%, 50%)`} 
                                strokeWidth={2} 
                                connectNulls={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                          <div className="text-center mt-2">
                            <strong>Mean {metricName}: {calculateMean(metricData, 'value')}</strong>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            </>
          )}
        </>
      )}
    </div>
  );
};

export default DashboardPage;