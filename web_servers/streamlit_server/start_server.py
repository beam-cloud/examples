"""
This script is used to run the streamlit server on Beam.

You can deploy this app by running `python start_server.py`
"""

from beam import Image, Pod

streamlit_server = Pod(
    image=Image().add_python_packages(["streamlit", "pandas", "altair"]),
    ports=[8501],  # Default port for streamlit
    cpu=1,
    memory=1024,
    entrypoint=["streamlit", "run", "app.py"],
)

# Create the pod on Beam
res = streamlit_server.create()

# Print the URL of the streamlit app
print("✨ Streamlit server hosted at:", res.url)
