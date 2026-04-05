# 🌿 PlantHealth AI - Hugging Face Spaces Deployment Guide

This guide will help you deploy your Plant Health AI application to Hugging Face Spaces, making it publicly accessible with a shareable URL.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Git LFS**: Install Git Large File Storage for the model file (94 MB)
3. **Hugging Face CLI** (optional): For easier deployment

## 🚀 Deployment Steps

### Step 1: Install Git LFS

**Windows:**
```bash
# Download from: https://git-lfs.github.com/
# Or use chocolatey:
choco install git-lfs
git lfs install
```

**Mac/Linux:**
```bash
# Mac
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Initialize
git lfs install
```

### Step 2: Create a New Space on Hugging Face

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Space name**: `plant-health-ai` (or your preferred name)
   - **License**: MIT
   - **Select SDK**: Docker
   - **Hardware**: CPU basic (free) or T4 small (GPU, paid but better performance)
3. Click **"Create Space"**

### Step 3: Set Up Git LFS for Model File

The model file (`plant_disease_model.pth`, 94 MB) needs to be tracked with Git LFS:

```bash
# In your project directory
cd E:\Projects\Plant-Health-AI

# Track .pth files with LFS (already configured in .gitattributes)
git lfs track "*.pth"

# Verify LFS is tracking the file
git lfs ls-files
```

### Step 4: Add Hugging Face Remote and Push

```bash
# Add Hugging Face Space as remote
# Replace YOUR_USERNAME with your actual Hugging Face username
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai

# Commit all deployment files
git add .gitattributes app.py Dockerfile requirements_hf.txt frontend/dist/
git commit -m "Add Hugging Face Spaces deployment configuration"

# Push to Hugging Face (you'll be prompted for credentials)
git push hf main

# If your branch is named differently (e.g., master):
git push hf master:main
```

### Step 5: Configure Space Settings (Optional)

After pushing, you can configure your Space:

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai`
2. Click **"Settings"** tab
3. Configure:
   - **Hardware**: Upgrade to T4 GPU for faster inference (optional, paid)
   - **Secrets**: Add any environment variables if needed
   - **Sleep time**: Configure auto-sleep (free tier sleeps after inactivity)

### Step 6: Wait for Build

- Hugging Face will automatically build your Docker container
- Check the **"Logs"** tab to monitor the build process
- Build typically takes 5-10 minutes
- Once complete, your Space will be live!

## 🌐 Your Public URL

After deployment, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai
```

Share this URL with anyone - no authentication required!

## 🔧 Alternative: Using Hugging Face CLI

Install the CLI for easier deployment:

```bash
# Install Hugging Face CLI
pip install huggingface-hub[cli]

# Login
huggingface-cli login

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai
cd plant-health-ai

# Copy files from your project
cp -r ../Plant-Health-AI/backend .
cp -r ../Plant-Health-AI/src .
cp -r ../Plant-Health-AI/frontend/dist ./frontend/
cp ../Plant-Health-AI/{app.py,Dockerfile,requirements_hf.txt,config.yaml} .

# Track model with LFS
git lfs track "*.pth"
git add .gitattributes

# Add and push
git add .
git commit -m "Initial deployment"
git push
```

## 📦 Files Required for Deployment

These files are now in your project root:

- ✅ `app.py` - Combined FastAPI + React app
- ✅ `Dockerfile` - Docker configuration
- ✅ `requirements_hf.txt` - Python dependencies
- ✅ `.gitattributes` - Git LFS configuration
- ✅ `frontend/dist/` - Built React app
- ✅ `backend/` - FastAPI backend
- ✅ `src/` - Core utilities (OOD detection, etc.)
- ✅ `backend/models/plant_disease_model.pth` - Trained model (94 MB)
- ✅ `backend/models/class_names.json` - Class labels

## 🎯 Testing Before Deployment

Test the Docker container locally:

```bash
# Build Docker image
docker build -t plant-health-ai .

# Run container
docker run -p 7860:7860 plant-health-ai

# Open browser
# http://localhost:7860
```

## 🐛 Troubleshooting

### Build Fails - Model File Too Large

**Solution**: Ensure Git LFS is installed and tracking the model:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add backend/models/plant_disease_model.pth
git commit --amend
git push hf main --force
```

### Frontend Not Loading

**Solution**: Rebuild frontend and commit:
```bash
cd frontend
npm run build
cd ..
git add frontend/dist/
git commit -m "Update frontend build"
git push hf main
```

### API Returns 500 Error

**Solution**: Check logs in Hugging Face Space:
1. Go to your Space
2. Click "Logs" tab
3. Look for error messages
4. Common issues:
   - Model file not found → Check Git LFS
   - Import errors → Check requirements_hf.txt
   - Path issues → Check app.py paths

### Space is Sleeping (Free Tier)

**Solution**: 
- Free tier Spaces sleep after 48 hours of inactivity
- First request will wake it up (takes ~30 seconds)
- Upgrade to paid tier for always-on hosting

## 💰 Cost Estimation

**Free Tier (CPU basic):**
- ✅ Perfect for demos and testing
- ✅ Unlimited public access
- ⚠️ Slower inference (~2-5 seconds)
- ⚠️ Auto-sleeps after inactivity

**Paid Tier (T4 GPU small - $0.60/hour):**
- ✅ Fast inference (~200ms)
- ✅ Always-on (no sleep)
- ✅ Better for production use
- 💵 ~$432/month if running 24/7

**Recommendation**: Start with free tier, upgrade if needed.

## 🔄 Updating Your Deployment

To update your deployed Space:

```bash
# Make changes to your code
# Rebuild frontend if needed
cd frontend && npm run build && cd ..

# Commit and push
git add .
git commit -m "Update: description of changes"
git push hf main
```

Hugging Face will automatically rebuild and redeploy.

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Git LFS Documentation](https://git-lfs.github.com/)

## 🎉 Next Steps

After deployment:

1. ✅ Test your public URL
2. ✅ Share with colleagues/friends
3. ✅ Monitor usage in Space analytics
4. ✅ Add Space to your portfolio/resume
5. ✅ Consider adding more features based on feedback

---

**Need Help?** 
- Check Hugging Face Community: [discuss.huggingface.co](https://discuss.huggingface.co)
- Open an issue in your GitHub repo
- Check the Space logs for error messages
