import os, sys, tempfile
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["DEMO_MODE"] = "true"
os.environ["DISABLE_POLLER"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
