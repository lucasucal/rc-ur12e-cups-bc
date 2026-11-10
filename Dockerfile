# Contestant image for a LeRobot ACT policy. FROM the same stack we trained with
# so inference matches training (docs/platform.md §6.1). On the Sparks this is
# the local `rc-lerobot:train` image; a team would FROM their cloud train image.
FROM rc-lerobot:train
# Committed train image still has ENTRYPOINT python3 from pip; clear it so CMD runs.
ENTRYPOINT []

WORKDIR /app
COPY rc_sdk ./rc_sdk
COPY policy.py obs.py manifest.json ./
RUN python3 -m pip install --no-cache-dir ./rc_sdk
# Optional locally-copied checkpoint (real teams mount weights at /weights).
COPY ckpt/ /app/ckpt/

CMD ["python3", "-m", "rc_sdk", "--manifest", "manifest.json", "--policy", "policy:ActPolicy"]
