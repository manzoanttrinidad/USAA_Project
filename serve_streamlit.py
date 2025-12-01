import shlex
import subprocess
from pathlib import Path
import os
import modal

# Path to your Streamlit script
streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "openai",
        "supabase",       # only if you still need it
        "python-dotenv",  # only if needed
    )
    # optional: force rebuild if you’re tweaking the image a lot
    # .env({"FORCE_REBUILD": "true"})
    # Copy your Streamlit app
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
    # Copy RAG core logic
    .add_local_file(Path(__file__).parent / "rag_core.py", "/root/rag_core.py")
    
)

# Name can be whatever you like
app = modal.App(name="fraud-rag-dashboard", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError("streamlit_run.py not found next to serve_streamlit_rag.py")


@app.function(
    secrets=[modal.Secret.from_name("my-secret")],
    allow_concurrent_inputs=100,
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))

    cmd = (
        f"streamlit run {target} "
        "--server.port 8000 "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false "
        "--server.address 0.0.0.0"
    )

    # Build environment variables, pulling from Modal secrets
    env_vars = {}

    # Optional Supabase pieces if you still use them
    if os.getenv("SUPABASE_KEY"):
        env_vars["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
    if os.getenv("SUPABASE_URL"):
        env_vars["SUPABASE_URL"] = os.getenv("SUPABASE_URL")

    # 🔑 Make sure OPENAI_API_KEY from the secret is available to Streamlit
    if os.getenv("OPENAI_API_KEY"):
        env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Include current environment for PATH etc.
    env_vars.update(os.environ)

    subprocess.Popen(cmd, shell=True, env=env_vars)
