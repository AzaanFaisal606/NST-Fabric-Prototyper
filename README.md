# NST Font Stylizer

Local web app that stylizes font glyphs using Neural Style Transfer.

## Run backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
