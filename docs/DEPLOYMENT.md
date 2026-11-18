# Deploying DocuMind

The app is one container: FastAPI serves both the API and the web page, so there is nothing else to host. It listens on `$PORT` (default `7860`) and exposes `/api/health` for health checks.

**Memory:** the two models need roughly 1 GB RAM. Hugging Face Spaces (16 GB free) is the easiest free option. On a 512 MB host, set `USE_RERANKER=false`.

**Secrets:** never commit `.env`. Add `GROQ_API_KEY` in the host's secret/environment settings.

---
## Option A – Hugging Face Spaces (free, recommended)

1. Create a free account at <https://huggingface.co> and click **New → Space**.
2. Name it (e.g. `documind`), choose **SDK: Docker**, hardware **CPU basic (free)**, visibility **Public**.
3. Clone the empty Space and copy the project into it:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/documind hf-documind
   cp -r documind/. hf-documind/        # copy everything including Dockerfile
   cd hf-documind
   ```
4. Put this block at the very **top of `README.md`** (Spaces reads it as configuration):
   ```yaml
   ---
   title: DocuMind
   emoji: 📄
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
5. In the Space: **Settings → Variables and secrets → New secret** → name `GROQ_API_KEY`, value = your key.
6. Commit and push (Hugging Face asks for an access token as the password; create one at *Settings → Access Tokens* with write permission):
   ```bash
   git add . && git commit -m "Deploy DocuMind" && git push
   ```
7. Watch the **Logs** tab. The first build takes ~8–12 minutes (it downloads PyTorch and the models). When it shows *Running*, your public link is `https://<your-username>-documind.hf.space`.

Free Spaces sleep after inactivity and the disk is wiped on restart: uploaded documents are temporary, which is fine for a demo.

---
## Option B – Render

1. Push the project to a GitHub repository.
2. In Render: **New → Blueprint**, select the repo (it reads `render.yaml`).
3. When prompted, enter `GROQ_API_KEY`.
4. Use the **Starter** plan (512 MB free tier is too small with the reranker; or set `USE_RERANKER=false`).
5. After deploy, open the `onrender.com` URL. Add a **persistent disk** mounted at `/home/user/app/data` if you want documents to survive restarts.

## Option C – Any Docker host (Railway, Fly.io, VPS)
```bash
docker build -t documind .
docker run -d -p 80:7860 -e GROQ_API_KEY=gsk_xxx -v documind-data:/home/user/app/data documind
```
Put it behind HTTPS (Caddy / nginx / the platform's built-in TLS).

---
## After deploying
- Open the link, upload a sample document, ask a question, and run **Evaluate quality** once to confirm everything works end to end.
- Add the live link, GitHub link and a screenshot of the evaluation tab to your resume and README.
- For a public demo, keep the default limits (`MAX_UPLOAD_MB`, `MAX_DOCS_PER_WORKSPACE`) and consider a Groq key used only for this project, so a leaked key can be revoked easily.
