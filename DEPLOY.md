# Deploying the presentation (all free tiers)

Two pieces, because Vercel only hosts static sites:

| What | Where | Why |
|---|---|---|
| **Live demo**: upload a video or pick a sample, get the QA report | **Hugging Face Spaces** (free CPU) | needs Python + ffmpeg + MediaPipe |
| **Presentation site**: problem, method, results, clickable reports, PDF | **Vercel** (or Netlify Drop) | static HTML |

Do them in this order: the site needs the demo URL and the GitHub URL.

---

## Step 1: Live demo on Hugging Face Spaces (~10 min, no git needed)

1. Sign up / log in at https://huggingface.co.
2. Go to https://huggingface.co/new-space:
   - Space name: `microdrama-qa`
   - SDK: **Streamlit**
   - Hardware: **CPU basic (free)**
   - Visibility: Public
   - Click **Create Space**.
3. In the new Space, open the **Files** tab → **Add file → Upload files**. Upload these from your `kuku-qa` folder
   (drag the folders in, they keep their structure):
   - `app.py`
   - `requirements.txt`
   - `packages.txt` (tells the Space to install ffmpeg)
   - the whole `qa` folder
   - the whole `real_inputs` folder (the 10 real AI clips with their `.txt` scripts; skip `_normalised`)
   - **Replace the Space's `README.md`** with `deploy/hf_space/README.md` (it holds the Space settings header).
   Commit.
4. Open the **App** tab. The first build takes 3–6 minutes (it installs ffmpeg and MediaPipe). Watch **Logs** if it fails.
5. Test: pick `r06_av_offset_300ms` and click **Run QA**: it should flag the audio delay.
6. Your demo URL is `https://huggingface.co/spaces/<your-username>/microdrama-qa`.

Notes: free Spaces sleep after ~48 h idle and wake on the next visit (about 1 minute). Analysis runs at roughly
real time, so an 8 s clip takes about 10–20 s.

## Step 2: Code on GitHub (~5 min)

1. Create an empty public repo at https://github.com/new, e.g. `kuku-microdrama-qa` (no README).
2. In PowerShell, from the `kuku-qa` folder:
   ```powershell
   git init
   git add .
   git commit -m "Pre-promo QA gate for AI microdramas"
   git branch -M main
   git remote add origin https://github.com/<your-username>/kuku-microdrama-qa.git
   git push -u origin main
   ```
   Copy your `real_inputs` folder in first: the 10 real clips (MIT, credited in `SOURCES.json`) and their scripts get
   committed, as do the real-clip reports in `outputs/real/` (your raw outputs). `.gitignore` keeps out the download cache. Do not add Kuku's input pack anywhere.

## Step 3: Build the site with your real reports

On your machine (where `outputs/real` exists), from the `kuku-qa` folder:
```powershell
python tools/build_site.py --github https://github.com/<your-username>/kuku-microdrama-qa --demo https://huggingface.co/spaces/<your-username>/microdrama-qa
start site\index.html
```
It prints `real reports: 10`. Check the page locally, then commit the site to GitHub:
```powershell
git add site outputs/real
git commit -m "Presentation site"
git push
```

## Step 4: Host the site on Vercel (~3 min)

**Option A: from GitHub (recommended, auto-updates on every push)**
1. Log in at https://vercel.com with your GitHub account → **Add New… → Project** → import `kuku-microdrama-qa`.
2. **Framework Preset: Other**. **Root Directory: `site`** (click Edit and pick the folder). Leave Build and Output
   settings empty.
3. **Deploy**. You get `https://kuku-microdrama-qa.vercel.app` (or similar).

**Option B: no GitHub, just drag and drop (Netlify Drop)**
1. Open https://app.netlify.com/drop, log in, and drag the `site` folder onto the page. You get a live URL immediately.

## What to submit

- Presentation site URL (Vercel): links to everything else
- Live demo URL (Hugging Face Space)
- GitHub repo URL (code + run steps in README + raw outputs in `outputs/`)
- `docs/solution_doc.pdf` (the 2-page doc)
