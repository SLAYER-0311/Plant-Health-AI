# 🚀 Quick Start: Deploy to Hugging Face Spaces

## The Fastest Way to Deploy (5 minutes)

### Prerequisites
1. **Hugging Face Account**: [Sign up here](https://huggingface.co/join)
2. **Git LFS**: [Download and install](https://git-lfs.github.com/)

### Automated Deployment

```bash
# Run the automated deployment script
python deploy_to_hf.py
```

The script will:
- ✅ Check Git LFS installation
- ✅ Build the React frontend
- ✅ Configure Git LFS for model files
- ✅ Set up Hugging Face remote
- ✅ Commit deployment files
- ✅ Push to Hugging Face Spaces

### Manual Deployment (If you prefer)

1. **Install Git LFS**
   ```bash
   # Windows
   choco install git-lfs
   
   # Mac
   brew install git-lfs
   
   # Linux
   sudo apt-get install git-lfs
   
   # Initialize
   git lfs install
   ```

2. **Build Frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

3. **Create Space on Hugging Face**
   - Go to: https://huggingface.co/new-space
   - Name: `plant-health-ai` (or your choice)
   - SDK: **Docker**
   - Hardware: CPU basic (free) or T4 small (GPU, $0.60/hour)
   - Click **Create Space**

4. **Set Up Git LFS**
   ```bash
   git lfs track "*.pth"
   git add .gitattributes
   ```

5. **Add Hugging Face Remote**
   ```bash
   # Replace YOUR_USERNAME with your Hugging Face username
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai
   ```

6. **Commit and Push**
   ```bash
   # Add deployment files
   git add app.py Dockerfile requirements_hf.txt frontend/dist/ backend/ src/ config.yaml
   
   # Commit
   git commit -m "Deploy to Hugging Face Spaces"
   
   # Push
   git push hf main
   ```

7. **Wait for Build** (5-10 minutes)
   - Monitor at: `https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai`
   - Click **"Logs"** tab to see build progress

8. **Done!** 🎉
   Your app will be live at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/plant-health-ai
   ```

## What Gets Deployed?

Your deployment includes:

- ✅ **Full-stack app**: React frontend + FastAPI backend
- ✅ **OOD Detection**: Rejects non-leaf images (green cloth, toys, etc.)
- ✅ **Plant Disease Classifier**: 38 disease classes across 14 plant types
- ✅ **GPU support** (if you choose T4 hardware)
- ✅ **Public URL**: Share with anyone

## Cost

| Tier | Hardware | Speed | Cost | Best For |
|------|----------|-------|------|----------|
| Free | CPU basic | ~2-5s | $0 | Demos, testing |
| Paid | T4 GPU | ~200ms | $0.60/hr | Production, fast inference |

**Free tier limitations:**
- Auto-sleeps after 48 hours of inactivity
- First request after sleep takes ~30s to wake up
- Still great for sharing with others!

## Troubleshooting

### "Git LFS not found"
```bash
# Install Git LFS first
# Windows: choco install git-lfs
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

git lfs install
```

### "Model file too large"
Your model (94MB) needs Git LFS:
```bash
git lfs track "*.pth"
git add .gitattributes backend/models/plant_disease_model.pth
git commit --amend
git push hf main --force
```

### "Frontend not loading"
Rebuild and push:
```bash
cd frontend && npm run build && cd ..
git add frontend/dist/
git commit -m "Update frontend"
git push hf main
```

### "Build failed"
Check logs at your Space → "Logs" tab. Common issues:
- Missing files → Check git add
- Import errors → Check requirements_hf.txt
- Model not found → Check Git LFS setup

## Next Steps

After deployment:

1. **Test your public URL**
2. **Share with others** (no auth required!)
3. **Monitor usage** in Space analytics
4. **Upgrade to GPU** if needed for faster inference
5. **Update anytime**: Just `git push hf main`

## Need Help?

- 📖 Full guide: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- 💬 Hugging Face Community: [discuss.huggingface.co](https://discuss.huggingface.co)
- 🐛 Issues: Check Space logs or GitHub issues

---

**Ready?** Run `python deploy_to_hf.py` to get started! 🚀
