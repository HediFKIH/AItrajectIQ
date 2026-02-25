# LLMInsight

LLMInsight is an **observability and evaluation platform for LLM-powered applications**.  
It helps you monitor interactions, track metadata, and evaluate responses against selected metrics — all in one place.

Instead of guessing how your model behaves in production, LLMInsight logs every interaction and continuously evaluates it.

---

## What It Does

- Logs every LLM interaction from your application
- Stores metadata (model, tokens, latency, tuning parameters, etc.)
- Evaluates responses using selected metrics
- Aggregates metrics over time
- Provides a dashboard to visualize performance per project

---

# Architecture Overview

## Backend (FastAPI)

- Handles user and project management
- Receives and stores interaction logs
- Evaluates interactions against selected metrics

## Frontend (React)

- Displays monitored projects
- Shows interaction logs and metadata
- Visualizes aggregated evaluation metrics

---

# Getting Started

## 1- Start the Backend

Navigate to the `app` directory:

```bash
cd app
uvicorn main:app
```

The API will typically be available at:

```
http://127.0.0.1:8000/docs
```

You can manage users and projects directly from the Swagger UI.

---

## 2- Create a User and Project

Using the Swagger API:

1. Create a **User**
2. Create a **Project**
   - Associate it with the `userId`
   - Pick the evaluation metrics you want applied to that project

---

## 3- Integrate LLMInsight into Your Application

Once your project is created:

- Retrieve the `projectId`
- Integrate it into your application

Your application should send interaction logs to LLMInsight using this `projectId`.

---

# Visualizing Data

## 1- Configure the Frontend

Open:

```
interface/src/ProjectsPage.jsx
```

Inject your `userId` in the appropriate location.

---

## 2- Run the Frontend

Navigate to the `interface` directory:

```bash
cd interface
npm run dev
```

The development server will start, and you’ll be able to access the UI locally.

---

# What You’ll See

## Projects List

All monitored projects associated with the `userId`.

## Interaction Logs (Per Project)

Each interaction includes metadata such as:

- Model used
- Number of tokens
- Latency
- Temperature and other tuning parameters
- Timestamp

## Metrics Dashboard

- Aggregated metrics over time
- Project-level performance tracking
- Trend visualization

---

# How Evaluation Works

- Metrics are selected at **project creation**
- Every interaction is evaluated only against relevant metrics
- Results are stored alongside the interaction log
- Aggregations are computed over time for dashboard visualization

---

# 🛠 Development Stack

- Backend: FastAPI + Uvicorn
- Frontend: React + npm
