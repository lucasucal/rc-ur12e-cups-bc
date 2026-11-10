# Contestant image: train and serve with the same Python deps.
# Base is public; torch is installed at build time so this builds on the Spark
# the same way a team would after training on it.
FROM python:3.12-slim-bookworm

WORKDIR /app
RUN pip install --no-cache-dir torch numpy pillow
COPY rc_sdk ./rc_sdk
COPY model.py policy.py obs.py manifest.json ./
RUN pip install --no-cache-dir ./rc_sdk

CMD ["python", "-m", "rc_sdk", "--manifest", "manifest.json", "--policy", "policy:CupBCPolicy"]
