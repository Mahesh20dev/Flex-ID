# Flex-ID Deployment & Sharing Guide

## 1. Quick Start (Local)
If you just want to run the project on your machine:

1.  **Build the Image**:
    ```bash
    docker build -t flex-id .
    ```
2.  **Run the Container**:
    ```bash
    docker run -p 5000:5000 flex-id
    ```
3.  **Open App**: Go to `http://localhost:5000`.

---

## 2. How to Share with a Friend
To let a friend run this on their system, they don't need to install Python or Node.js. They only need **Docker Desktop**.

### Step A: You (The Owner)
1.  Ensure your code is pushed to GitHub:
    ```bash
    git push origin main
    ```
2.  Send your **GitHub Repository URL** to your friend.

### Step B: Your Friend
1.  **Install Docker Desktop**: Download and install from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  **Clone the Repo**:
    ```bash
    git clone <YOUR_GITHUB_REPO_URL>
    cd 7SemProject
    ```
3.  **Build & Run**:
    ```bash
    docker build -t flex-id .
    docker run -p 5000:5000 flex-id
    ```
4.  That's it! They can now access the full app at `http://localhost:5000`.

---

## 3. Cloud Deployment (Render)
To make it accessible via a public URL (e.g., `https://flex-id.onrender.com`):

1.  Go to [render.com](https://render.com).
2.  Create a **New Web Service**.
3.  Connect your GitHub repository.
4.  Select **Docker** as the Runtime.
5.  Click **Deploy**.

**Note on Data:**
For cloud deployment, if you want to keep the training history even after the server restarts, you must upgrade to a paid plan and add a "Persistent Disk". On the free plan, data wipes on restart.
